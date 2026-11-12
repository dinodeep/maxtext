# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests that gpu_collective_overlap_pass keeps te_ep (EP) async windows wide:
latency lookup for wrapped custom-calls, the window guard on operand-chain
moves and the start-blocker chase, and fill pins across iterations.

Uses small fake HLO objects (the pass only needs name/opcode/operands/users/
to_string and the computation's instructions)."""

# pylint: disable=protected-access

import re
import unittest
from unittest import mock

from maxtext.utils import gpu_collective_overlap_pass as pass_lib

_FP = "d2f455245c5a61520dcabdcbc6a7ef38"


class _Opcode:

  def __init__(self, opcode: str):
    # "custom-call" -> "kCustomCall", as HloOpcode enum names print.
    self.name = "k" + "".join(part.capitalize() for part in opcode.split("-"))


class FakeInst:
  """Just enough of an HloInstruction for the pass."""

  def __init__(self, name, opcode, operands=(), attrs="", text_opcode=None, wrapped_root=None):  # pylint: disable=too-many-positional-arguments
    self.name = name
    self.opcode = _Opcode(opcode)
    self._operands = list(operands)
    self._users: list = []
    self._attrs = attrs
    self._text_opcode = text_opcode or opcode
    self._wrapped_root = wrapped_root
    for op in self._operands:
      op._users.append(self)

  def operands(self):
    return list(self._operands)

  def users(self):
    return list(self._users)

  def async_wrapped_root(self):
    if self._wrapped_root is None:
      raise ValueError("not an async op")
    return self._wrapped_root

  def to_string(self):
    ops = ", ".join(f"%{o.name}" for o in self._operands)
    attrs = f", {self._attrs}" if self._attrs else ""
    return f"%{self.name} = f32[] {self._text_opcode}({ops}){attrs}"

  def __repr__(self):
    return self.name


class FakeComp:
  """Just enough of an HloComputation for the pass."""

  def __init__(self, name, insts, root=None):
    self.name = name
    self._insts = list(insts)
    self._root = root if root is not None else self._insts[-1]

  def instructions(self):
    return list(self._insts)

  def root_instruction(self):
    return self._root


class FakeSchedule:

  def __init__(self):
    self.sequences = {}

  def set_sequence(self, comp, seq):
    self.sequences[comp.name] = list(seq)


class _Builder:
  """Builds a computation: heavy/blocker ops are fusions with a dot (not
  trivially movable), chain ops are fusions with only converts (trivially
  movable), te_ep calls are async call-starts wrapping a custom-call."""

  def __init__(self):
    self.comp_by_name: dict = {}
    self.costs: dict = {}
    self._dot_body = self._comp("heavy_body", [FakeInst("dot.0", "dot")])
    self._trivial_body = self._comp("trivial_body", [FakeInst("convert.0", "convert")])

  def _comp(self, name, insts, root=None):
    comp = FakeComp(name, insts, root)
    self.comp_by_name[name] = comp
    return comp

  def heavy(self, name, cost, operands=()):
    self.costs[name] = cost
    return FakeInst(name, "fusion", operands, attrs="kind=kLoop, calls=%heavy_body")

  def trivial(self, name, cost, operands=()):
    self.costs[name] = cost
    return FakeInst(name, "fusion", operands, attrs="kind=kLoop, calls=%trivial_body")

  def te_ep(self, idx, target, latency, operands=()):
    """(call-start, call-done) of a te_ep custom-call with `latency`."""
    cc = FakeInst(f"{target}.{idx}", "custom-call", attrs=f'custom_call_target="{target}"')
    self._comp(f"compute_on.{idx}", [cc])
    self.costs[cc.name] = latency
    start = FakeInst(
        f"call-start.{idx}",
        "async-start",
        operands,
        attrs=f'async_execution_thread="parallel", to_apply=%compute_on.{idx}',
        text_opcode="call-start",
    )
    return start, FakeInst(f"call-done.{idx}", "async-done", [start], text_opcode="call-done")

  def all_gather(self, idx, latency, operands=()):
    root = FakeInst(f"all-gather.{idx}", "all-gather")
    self.costs[root.name] = latency
    start = FakeInst(f"all-gather-start.{idx}", "all-gather-start", operands, wrapped_root=root)
    return start, FakeInst(f"all-gather-done.{idx}", "all-gather-done", [start])


def _start_of_done(seq):
  return {u: s for s in seq if pass_lib._is_async_start(s) for u in s.users()}


def _window_overlap(seq, start, done, costs):
  i, j = seq.index(start), seq.index(done)
  return sum(costs.get(x.name, 0.0) for x in seq[i + 1 : j])


class _PassTestCase(unittest.TestCase):
  """Isolates the module-level cost and pin state of the pass per test."""

  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.object(pass_lib, "_pgle_costs_by_fingerprint", {}))
    self.enterContext(mock.patch.object(pass_lib, "_profile_costs", {}))
    self.enterContext(mock.patch.object(pass_lib, "_inst_cost_cache", {}))
    self.enterContext(mock.patch.object(pass_lib, "_te_ep_overrides", {}))
    self.enterContext(mock.patch.object(pass_lib, "_host_offload_copy_names", set()))
    self.enterContext(mock.patch.object(pass_lib, "_window_fill_pins", {}))
    self.enterContext(mock.patch.object(pass_lib, "_wrapped_custom_call_cache", {}))
    self.enterContext(mock.patch.object(pass_lib, "_WINDOW_GUARD", True))
    self.enterContext(mock.patch.object(pass_lib, "_WINDOW_GUARD_MODE", "total"))
    self.b = _Builder()

  def load_costs(self):
    """Pool the builder's costs as fingerprinted PGLE costs and select them."""
    pass_lib._pool_costs({f"{_FP}::{name}": cost for name, cost in self.b.costs.items()})
    pass_lib._select_module_costs(_FP, "test")


class WrappedCustomCallKeyTest(_PassTestCase):
  """Fix 1: a te_ep_prepare call-start's wrapped root is a tuple of
  get-tuple-elements; its latency is keyed by the custom-call's name."""

  def _prepare_call_start(self):
    param = FakeInst("param.608", "parameter")
    cc = FakeInst("te_ep_prepare_ffi.78", "custom-call", [param], attrs='custom_call_target="te_ep_prepare_ffi"')
    gte0 = FakeInst("get-tuple-element.16106.0", "get-tuple-element", [cc], attrs="index=0")
    gte2 = FakeInst("get-tuple-element.16107.0", "get-tuple-element", [cc], attrs="index=2")
    root = FakeInst("tuple.1010.0", "tuple", [gte0, gte2])
    self.b._comp("compute_on_99.177.58_spmd.clone.9", [param, cc, gte0, gte2, root], root)
    self.b.costs["te_ep_prepare_ffi.78"] = 241.0
    arg = FakeInst("bitcast.973.0", "bitcast")
    return FakeInst(
        "call-start.98",
        "async-start",
        [arg],
        attrs='async_execution_thread="parallel", to_apply=%compute_on_99.177.58_spmd.clone.9',
        text_opcode="call-start",
    )

  def test_async_custom_call_start_resolves_to_wrapped_custom_call(self):
    start = self._prepare_call_start()
    self.load_costs()
    self.assertEqual(pass_lib._resolve_profile_key(start, self.b.comp_by_name), "te_ep_prepare_ffi.78")
    # The latency is found, so the window is protected and counted.
    self.assertEqual(pass_lib._profile_costs.get(pass_lib._resolve_profile_key(start, self.b.comp_by_name)), 241.0)

  def test_live_xspace_correction_key_matches(self):
    start = self._prepare_call_start()
    pass_lib._te_ep_overrides["te_ep_prepare_ffi.78"] = 300.0
    self.load_costs()
    self.assertEqual(pass_lib._profile_costs[pass_lib._resolve_profile_key(start, self.b.comp_by_name)], 300.0)

  def test_wrapped_key_cache_is_per_module(self):
    start = self._prepare_call_start()
    self.load_costs()
    self.assertEqual(pass_lib._resolve_profile_key(start, self.b.comp_by_name), "te_ep_prepare_ffi.78")
    self.assertIn("call-start.98", pass_lib._wrapped_custom_call_cache)
    pass_lib._select_module_costs(_FP, "next module")
    self.assertEqual(pass_lib._wrapped_custom_call_cache, {})

  def test_toggle_off_keeps_old_key(self):
    start = self._prepare_call_start()
    self.load_costs()
    with mock.patch.object(pass_lib, "_RESOLVE_WRAPPED_CUSTOM_CALL", False):
      self.assertEqual(pass_lib._resolve_profile_key(start, self.b.comp_by_name), "tuple.1010.0")

  def test_custom_call_root_and_collectives_unchanged(self):
    start, _ = self.b.te_ep(99, "te_ep_dispatch_ffi", 4364.4)
    ag, _ = self.b.all_gather(1, 200.0)
    self.load_costs()
    self.assertEqual(pass_lib._resolve_profile_key(start, self.b.comp_by_name), "te_ep_dispatch_ffi.99")
    self.assertEqual(pass_lib._resolve_profile_key(ag, self.b.comp_by_name), "all-gather.1")
    self.assertTrue(pass_lib._is_te_ep_call_start(start, self.b.comp_by_name))

  def test_wrapper_without_custom_call_keeps_old_key(self):
    inner = FakeInst("add.1", "add")
    self.b._comp("passthrough", [inner])
    start = FakeInst("async-start.5", "async-start", attrs="calls=%passthrough")
    self.load_costs()
    with mock.patch.object(pass_lib, "_RESOLVE_WRAPPED_CUSTOM_CALL", False):
      old_key = pass_lib._resolve_profile_key(start, self.b.comp_by_name)
    self.assertEqual(pass_lib._resolve_profile_key(start, self.b.comp_by_name), old_key)


class OperandChainGuardTest(_PassTestCase):
  """Fix 2: the fill margin check sees only the candidate's own cost at its own
  position; the trivial operand chain moved with it can leave an EP window."""

  def _build(self):
    """EP window holding a trivial chain op (80 us) and filler (30 us) vs a 100 us
    latency, an exposed all-gather after it, and a 150 us candidate whose
    operand chain is that chain op."""
    b = self.b
    prod = b.heavy("prod.0", 0.0)
    ep_start, ep_done = b.te_ep(99, "te_ep_dispatch_ffi", 100.0, [prod])
    chain = b.trivial("loop_convert_fusion.1", 80.0, [prod])
    filler = b.heavy("filler.0", 30.0, [prod])
    ag_start, ag_done = b.all_gather(1, 200.0, [prod])
    cand = b.heavy("candidate.0", 150.0, [chain])
    seq = [prod, ep_start, chain, filler, ep_done, ag_start, ag_done, cand]
    self.load_costs()
    return seq, ep_start, ep_done, chain, cand, ag_start

  def _fill(self, seq, guard):
    sod = _start_of_done(seq)
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    comp = FakeComp("region_161.228_spmd.1", seq)
    _, new_seq, _, _ = pass_lib._fill_exposed_collectives_with_heavy_compute(
        list(seq), FakeSchedule(), comp, sod, positions, names, self.b.comp_by_name, "test", guard=guard
    )
    return new_seq

  def test_unguarded_fill_narrows_ep_window(self):
    seq, ep_start, ep_done, chain, cand, ag_start = self._build()
    new_seq = self._fill(seq, guard=None)
    # The old behavior this guards against: the chain left the EP window.
    self.assertGreater(new_seq.index(cand), new_seq.index(ag_start))
    self.assertGreater(new_seq.index(chain), new_seq.index(ep_done))
    self.assertLess(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)

  def test_window_mode_rejects_operand_chain_move(self):
    seq, ep_start, ep_done, chain, cand, _ = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    with mock.patch.object(pass_lib, "_WINDOW_GUARD_MODE", "window"):
      new_seq = self._fill(seq, guard)
    self.assertEqual(new_seq, seq)
    self.assertLess(new_seq.index(chain), new_seq.index(ep_done))
    self.assertGreaterEqual(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)
    self.assertGreaterEqual(guard.rejections, 1)
    del cand

  def test_total_mode_allows_net_gain(self):
    # The EP window loses 70 us of hiding but the all-gather window gains 200.
    seq, ep_start, ep_done, chain, cand, ag_start = self._build()
    # A free op in the all-gather window: the fill inserts at the target's old
    # index after dropping the chain, so it lands one slot later.
    seq.insert(seq.index(ag_start) + 1, self.b.heavy("ag_pad.0", 0.0))
    self.load_costs()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    new_seq = self._fill(seq, guard)
    self.assertGreater(new_seq.index(cand), new_seq.index(ag_start))
    self.assertGreater(new_seq.index(chain), new_seq.index(ep_done))
    self.assertLess(new_seq.index(cand), new_seq.index(seq[seq.index(ag_start) + 2]))  # before ag done
    self.assertEqual(guard.rejections, 0)
    self.assertLess(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)

  def test_total_mode_rejects_net_loss(self):
    # Same move, but the all-gather is already almost hidden: it gains 20 us
    # of hiding while the EP window loses 70.
    seq, ep_start, ep_done, chain, _, ag_start = self._build()
    ag_done = seq[seq.index(ag_start) + 1]
    seq.insert(seq.index(ag_done), self.b.heavy("ag_filler.0", 180.0))
    self.load_costs()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    new_seq = self._fill(seq, guard)
    self.assertLess(new_seq.index(chain), new_seq.index(ep_done))
    self.assertGreaterEqual(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)
    self.assertGreaterEqual(guard.rejections, 1)

  def test_total_mode_counts_every_window(self):
    seq, ep_start, _, chain, cand, ag_start = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    ag_done = seq[seq.index(ag_start) + 1]
    # chain + candidate into the all-gather window: EP deficit 0 -> 70, all-gather 200 -> 0.
    moved = [x for x in seq if x not in (chain, cand)]
    moved[moved.index(ag_done) : moved.index(ag_done)] = [chain, cand]
    self.assertIsNone(guard.violation(seq, moved))
    with mock.patch.object(pass_lib, "_WINDOW_GUARD_MODE", "window"):
      self.assertIn("call-start.99", guard.violation(seq, moved))
    # The reverse move grows the total deficit by 130 us.
    reason = guard.violation(moved, seq)
    self.assertIn("total window deficit would grow 70.0 -> 200.0", reason)
    # An exempt window is left out of the totals.
    self.assertIsNone(guard.violation(moved, seq, exempt=(ag_start,)))
    del ep_start

  def test_guard_allows_move_that_keeps_latency_hidden(self):
    seq, ep_start, ep_done, _, cand, ag_start = self._build()
    # With enough other compute in the EP window, taking the chain out is fine.
    extra = self.b.heavy("filler.1", 90.0)
    seq.insert(seq.index(ep_done), extra)
    self.load_costs()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    new_seq = self._fill(seq, guard)
    self.assertGreater(new_seq.index(cand), new_seq.index(ag_start))
    self.assertGreaterEqual(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)

  def test_guard_toggle_off(self):
    seq, ep_start, ep_done, _, _, _ = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    with mock.patch.object(pass_lib, "_WINDOW_GUARD", False):
      new_seq = self._fill(seq, guard)
    self.assertLess(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)


class ConsumerFirstFillTest(_PassTestCase):
  """A te_ep_prepare window feeds its te_ep_dispatch start; filled first, it
  takes the compute the (much longer) dispatch window needed."""

  def _build(self):
    b = self.b
    prep_start, prep_done = b.te_ep(146, "te_ep_prepare_ffi", 50.0)
    gte = FakeInst("get-tuple-element.1", "get-tuple-element", [prep_done], attrs="index=1")
    disp_start, disp_done = b.te_ep(147, "te_ep_dispatch_ffi", 300.0, [gte])
    cand = b.heavy("te_grouped_quantize_ffi.1251", 400.0)
    seq = [prep_start, prep_done, gte, disp_start, disp_done, cand]
    self.load_costs()
    return seq, disp_start, disp_done, prep_start, prep_done

  def _fill(self, seq):
    """Guarded fill of `seq`; returns the new sequence."""
    sod = _start_of_done(seq)
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    guard = pass_lib._WindowGuard(sod, self.b.comp_by_name, "region_332.420_spmd.1")
    _, new_seq, _, _ = pass_lib._fill_exposed_collectives_with_heavy_compute(
        list(seq),
        FakeSchedule(),
        FakeComp("region_332.420_spmd.1", seq),
        sod,
        positions,
        names,
        self.b.comp_by_name,
        "test",
        guard=guard,
    )
    return new_seq

  def test_done_feeds_start(self):
    seq, disp_start, _, prep_start, prep_done = self._build()
    self.assertTrue(pass_lib._done_feeds_start(prep_done, disp_start))
    self.assertFalse(pass_lib._done_feeds_start(prep_done, prep_start))
    del seq

  def test_dispatch_window_is_filled_first(self):
    seq, disp_start, disp_done, _, _ = self._build()
    with mock.patch.object(pass_lib, "_FILL_CONSUMER_FIRST", True):
      new_seq = self._fill(seq)
    self.assertGreaterEqual(_window_overlap(new_seq, disp_start, disp_done, self.b.costs), 300.0)

  def test_start_order_toggle(self):
    seq, disp_start, disp_done, prep_start, prep_done = self._build()
    with mock.patch.object(pass_lib, "_FILL_CONSUMER_FIRST", False):
      new_seq = self._fill(seq)
    self.assertEqual(_window_overlap(new_seq, disp_start, disp_done, self.b.costs), 0.0)
    self.assertGreaterEqual(_window_overlap(new_seq, prep_start, prep_done, self.b.costs), 50.0)


class ExposedCollectiveChaseOrderTest(_PassTestCase):
  """The exposed-collective chase capped the chased start at the window's done
  even when the start reads that done (te_ep_prepare -> te_ep_dispatch),
  producing a sequence with the dispatch start before the prepare done."""

  def test_chase_never_moves_a_start_above_its_operand(self):
    b = self.b
    prep_start, prep_done = b.te_ep(146, "te_ep_prepare_ffi", 50.0)
    gte = FakeInst("get-tuple-element.1", "get-tuple-element", [prep_done], attrs="index=1")
    filler = b.heavy("filler.0", 10.0)
    disp_start, disp_done = b.te_ep(147, "te_ep_dispatch_ffi", 300.0, [gte])
    seq = [prep_start, prep_done, filler, gte, disp_start, disp_done]
    self.load_costs()
    sod = _start_of_done(seq)
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    moved, new_seq, _, _ = pass_lib._chase_exposed_collective_blocker_earlier(
        disp_done,
        sod,
        {disp_done, prep_done},
        positions[prep_start],
        positions[prep_done],
        list(seq),
        FakeSchedule(),
        FakeComp("region_332.420_spmd.1", seq),
        positions,
        names,
        self.b.comp_by_name,
    )
    self.assertFalse(moved)
    self.assertLess(new_seq.index(prep_done), new_seq.index(disp_start))


class StartBlockerChaseGuardTest(_PassTestCase):
  """Fix 2: the start-blocker chase moved an async start's producer to its
  floor with no check, out of a te_ep dispatch window."""

  def _build(self):
    b = self.b
    prod = b.heavy("prod.0", 0.0)
    ep_start, ep_done = b.te_ep(99, "te_ep_dispatch_ffi", 100.0, [prod])
    blocker = b.heavy("te_dbias_quantize_fusion.1", 80.0, [prod])
    filler = b.heavy("filler.0", 30.0, [prod])
    ag_start, ag_done = b.all_gather(1, 200.0, [blocker])
    seq = [prod, ep_start, blocker, filler, ep_done, ag_start, ag_done]
    self.load_costs()
    return seq, ep_start, ep_done, blocker, ag_start

  def _chase(self, seq, ag_start, guard):
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    comp = FakeComp("region_161.228_spmd.1", seq)
    changed, new_seq, *_ = pass_lib._chase_async_start_blocker_toward_heavy_compute(
        ag_start, comp, FakeSchedule(), list(seq), positions, names, self.b.comp_by_name, 8, "test", guard
    )
    return changed, new_seq

  def test_unguarded_chase_narrows_ep_window(self):
    seq, ep_start, ep_done, blocker, ag_start = self._build()
    changed, new_seq = self._chase(seq, ag_start, None)
    self.assertTrue(changed)
    self.assertLess(new_seq.index(blocker), new_seq.index(ep_start))
    self.assertLess(_window_overlap(new_seq, ep_start, ep_done, self.b.costs), 100.0)

  def test_guard_rejects_chase_hop(self):
    seq, ep_start, ep_done, blocker, ag_start = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "region_161.228_spmd.1")
    changed, new_seq = self._chase(seq, ag_start, guard)
    self.assertFalse(changed)
    self.assertEqual(new_seq, seq)
    self.assertGreater(new_seq.index(blocker), new_seq.index(ep_start))
    self.assertEqual(guard.rejections, 1)
    self.assertIsNotNone(re.search(r"call-start\.99", guard.violation(seq, pass_lib._reinsert(seq, [], 1, blocker))))
    del ep_done


class FillPinTest(_PassTestCase):
  """Fix 3: a widening relocation into an under-filled window is pinned there
  for later iterations, even where the slack alone would let it go."""

  COMP = "region_161.228_spmd.1"

  def _build(self):
    b = self.b
    prod = b.heavy("prod.0", 0.0)
    ep_start, ep_done = b.te_ep(110, "te_ep_dispatch_ffi", 100.0, [prod])
    resident = b.heavy("resident.0", 60.0, [prod])
    fill = b.heavy("te_grouped_quantize_ffi.1", 50.0, [prod])
    seq = [prod, ep_start, resident, ep_done, fill]
    self.load_costs()
    return seq, ep_start, ep_done, fill

  def _iteration_1(self, seq):
    """One fill (via the best-of driver) on a fresh guard, as _reorder_collectives does."""
    sod = _start_of_done(seq)
    guard = pass_lib._WindowGuard(sod, self.b.comp_by_name, self.COMP)
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    with mock.patch.object(pass_lib, "_fill_strategy_cache", {}):
      _, new_seq, _, _ = pass_lib._fill_exposed_collectives_best_of(
          list(seq),
          FakeSchedule(),
          FakeComp(self.COMP, seq),
          sod,
          positions,
          names,
          self.b.comp_by_name,
          "test",
          guard=guard,
      )
    return new_seq

  def _iteration_2_move_out(self, seq, ep_start, ep_done, fill):
    """Iteration 2: more compute landed in the window, so the slack would let
    the earlier fill go; returns the guard's verdict on moving it out."""
    later = self.b.heavy("later.0", 80.0)
    self.load_costs()
    seq = list(seq)
    seq.insert(seq.index(ep_done), later)
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, self.COMP)
    moved_out = pass_lib._reinsert(seq, [], seq.index(ep_start), fill)
    self.assertGreaterEqual(_window_overlap(moved_out, ep_start, ep_done, self.b.costs), 100.0)
    return guard.violation(seq, moved_out)

  def test_widening_relocation_persists_across_iterations(self):
    with mock.patch.object(pass_lib, "_PIN_WINDOW_FILLS", True):
      seq, ep_start, ep_done, fill = self._build()
      new_seq = self._iteration_1(seq)
      self.assertTrue(new_seq.index(ep_start) < new_seq.index(fill) < new_seq.index(ep_done))
      self.assertEqual(pass_lib._window_fill_pins[self.COMP], {fill.name: ep_start.name})
      reason = self._iteration_2_move_out(new_seq, ep_start, ep_done, fill)
    self.assertIsNotNone(reason)
    self.assertIn("pinned", reason)

  def test_without_pinning_slack_allows_undoing_the_fill(self):
    seq, ep_start, ep_done, fill = self._build()
    new_seq = self._iteration_1(seq)
    self.assertTrue(new_seq.index(ep_start) < new_seq.index(fill) < new_seq.index(ep_done))
    self.assertEqual(pass_lib._window_fill_pins.get(self.COMP, {}), {})
    self.assertIsNone(self._iteration_2_move_out(new_seq, ep_start, ep_done, fill))


_REAL_HLO = """HloModule consumer_first, is_scheduled=true

%prep (p0: f32[8]) -> (f32[8], f32[8]) {
  %p0 = f32[8] parameter(0)
  %te_ep_prepare_ffi.1 = (f32[8], f32[8]) custom-call(%p0), custom_call_target="te_ep_prepare_ffi"
  %g0 = f32[8] get-tuple-element(%te_ep_prepare_ffi.1), index=0
  %g1 = f32[8] get-tuple-element(%te_ep_prepare_ffi.1), index=1
  ROOT %t = (f32[8], f32[8]) tuple(%g0, %g1)
}

%disp (q0: f32[8]) -> f32[8] {
  %q0 = f32[8] parameter(0)
  ROOT %te_ep_dispatch_ffi.2 = f32[8] custom-call(%q0), custom_call_target="te_ep_dispatch_ffi"
}

ENTRY %main (x: f32[8]) -> f32[8] {
  %x = f32[8] parameter(0)
  %call-start.1 = ((f32[8]), (f32[8], f32[8])) call-start(%x), async_execution_thread="parallel", to_apply=%prep
  %call-done.1 = (f32[8], f32[8]) call-done(%call-start.1)
  %get-tuple-element.1 = f32[8] get-tuple-element(%call-done.1), index=0
  %bitcast.1 = f32[8] bitcast(%get-tuple-element.1)
  %call-start.2 = ((f32[8]), f32[8]) call-start(%bitcast.1), async_execution_thread="parallel", to_apply=%disp
  %call-done.2 = f32[8] call-done(%call-start.2)
  ROOT %add.1 = f32[8] add(%call-done.2, %x)
}
"""


class RealHloConsumerFirstTest(unittest.TestCase):
  """_done_feeds_start on instructions from jaxlib's HLO parser, whose Python
  wrappers are not identity-stable (a fresh object per access): an `is`
  comparison never matched, so FILL_CONSUMER_FIRST was a no-op on real HLO."""

  def setUp(self):
    super().setUp()
    from jax._src.lib import hlo as _hlo  # pylint: disable=import-outside-toplevel

    self.module = _hlo.hlo_module_from_text(_REAL_HLO)
    self.comp = next(c for c in self.module.computations() if c.name == "main")
    self.seq = list(self.module.schedule().sequence(self.comp))
    by_name = {inst.name: inst for inst in self.seq}
    self.prep = (by_name["call-start.1"], by_name["call-done.1"])
    self.disp = (by_name["call-start.2"], by_name["call-done.2"])

  def test_wrappers_are_not_identity_stable(self):
    # The premise: the same instruction reached two ways is two objects.
    via_operands = self.disp[0].operands()[0].operands()[0].operands()[0]
    self.assertEqual(via_operands.name, "call-done.1")
    self.assertIsNot(via_operands, self.prep[1])
    self.assertEqual(via_operands, self.prep[1])

  def test_done_feeds_start(self):
    self.assertTrue(pass_lib._done_feeds_start(self.prep[1], self.disp[0]))
    self.assertFalse(pass_lib._done_feeds_start(self.disp[1], self.prep[0]))

  def test_consumer_window_is_ordered_first(self):
    windows = [{"start": self.prep[0], "done": self.prep[1]}, {"start": self.disp[0], "done": self.disp[1]}]
    ordered = pass_lib._order_consumer_windows_first(windows)
    self.assertEqual([w["start"].name for w in ordered], ["call-start.2", "call-start.1"])


class EnvFlagTest(unittest.TestCase):

  def _flag(self, value, default):
    env = {} if value is None else {"COLLECTIVE_OVERLAP_TEST_FLAG": value}
    with mock.patch.dict("os.environ", env, clear=False):
      if value is None:
        import os  # pylint: disable=import-outside-toplevel

        os.environ.pop("COLLECTIVE_OVERLAP_TEST_FLAG", None)
      return pass_lib._env_flag("COLLECTIVE_OVERLAP_TEST_FLAG", default)

  def test_true_values(self):
    for v in ("1", "true", "TRUE", "Yes", "on", " On "):
      self.assertTrue(self._flag(v, False), v)

  def test_false_values(self):
    for v in ("0", "false", "False", "NO", "off", " off "):
      self.assertFalse(self._flag(v, True), v)

  def test_default_otherwise(self):
    for v in (None, "", "2", "maybe"):
      self.assertTrue(self._flag(v, True), v)
      self.assertFalse(self._flag(v, False), v)


class CostCacheTest(_PassTestCase):
  """_opcode_str and _resolve_inst_cost caches."""

  def test_opcode_str_cached_by_opcode_name(self):
    a = FakeInst("a", "get-tuple-element")
    self.assertEqual(pass_lib._opcode_str(a), "get-tuple-element")
    self.assertEqual(pass_lib._opcode_str_cache["kGetTupleElement"], "get-tuple-element")
    self.assertEqual(pass_lib._opcode_str(FakeInst("b", "custom-call")), "custom-call")

  def test_inst_cost_cached_and_cleared_on_cost_change(self):
    x = self.b.heavy("fusion.7", 10.0)
    self.load_costs()
    self.assertEqual(pass_lib._resolve_inst_cost(x, self.b.comp_by_name), 10.0)
    self.assertEqual(pass_lib._inst_cost_cache["fusion.7"], 10.0)
    # A direct edit is not seen (cached) ...
    pass_lib._profile_costs["fusion.7"] = 20.0
    self.assertEqual(pass_lib._resolve_inst_cost(x, self.b.comp_by_name), 10.0)
    # ... but selecting costs again clears the cache.
    self.b.costs["fusion.7"] = 30.0
    self.load_costs()
    self.assertEqual(pass_lib._resolve_inst_cost(x, self.b.comp_by_name), 30.0)

  def test_inst_cost_cleared_by_host_offload_registration(self):
    x = self.b.heavy("fusion.8", 10.0)
    self.load_costs()
    self.assertEqual(pass_lib._resolve_inst_cost(x, self.b.comp_by_name), 10.0)

    class _EmptyModule:
      name = "m"

      def computations(self):
        return []

      def make_nonfusion_computations(self):
        return []

    pass_lib._register_host_offload_copies(_EmptyModule())
    self.assertEqual(pass_lib._inst_cost_cache, {})


class DoneRelocationGuardTest(_PassTestCase):
  """_relocate_async_done_before_te_ep goes through the window guard too."""

  def _build(self):
    b = self.b
    ag_start, ag_done = b.all_gather(1, 100.0)
    f1 = b.heavy("f1.0", 120.0)
    ep_start, ep_done = b.te_ep(5, "te_ep_dispatch_ffi", 50.0)
    f2 = b.heavy("f2.0", 60.0)
    seq = [ag_start, f1, ep_start, f2, ag_done, ep_done]
    self.load_costs()
    return seq, ag_start, ag_done

  def _relocate(self, seq, ag_start, ag_done, guard):
    """(moved, new_seq) of the done relocation for the all-gather (latency 100 us)."""
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    moved, new_seq, _, _ = pass_lib._relocate_async_done_before_te_ep(
        ag_start,
        ag_done,
        FakeComp("c", seq),
        FakeSchedule(),
        list(seq),
        positions,
        names,
        self.b.comp_by_name,
        100.0,
        "test",
        guard,
    )
    return moved, new_seq

  def test_allowed_when_latency_stays_hidden(self):
    seq, ag_start, ag_done = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "c")
    moved, new_seq = self._relocate(seq, ag_start, ag_done, guard)
    self.assertTrue(moved)
    self.assertEqual(new_seq.index(ag_done), 2)

  def test_rejected_below_margin(self):
    seq, ag_start, ag_done = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "c")
    with mock.patch.object(pass_lib, "_WINDOW_GUARD_MARGIN_US", 50.0):
      moved, new_seq = self._relocate(seq, ag_start, ag_done, guard)
    self.assertFalse(moved)
    self.assertEqual(new_seq, seq)
    self.assertEqual(guard.rejections, 1)


class BlockerPath1ExemptionTest(_PassTestCase):
  """A path-1 blocker relocation already judged total exposed time, crediting
  the async start hoist it enables; the guard (one step only) would see just
  the cost. Exempt in "total" mode, guarded in "window" mode."""

  def _build(self):
    """`blocker` sits in a 100 us window (filled 140 us) right before the start
    of a 1000 us collective it feeds, whose window holds 200 us."""
    b = self.b
    prod = b.heavy("prod.0", 0.0)
    w2_start, w2_done = b.te_ep(108, "te_ep_dispatch_bwd_ffi", 100.0, [prod])
    f = b.heavy("filler.0", 60.0, [prod])
    blocker = b.heavy("input_reduce_fusion.355", 80.0, [prod])
    a_start, a_done = b.te_ep(110, "te_ep_dispatch_ffi", 1000.0, [blocker])
    x = b.heavy("gemm.0", 200.0, [prod])
    seq = [prod, w2_start, f, blocker, w2_done, a_start, x, a_done]
    self.load_costs()
    return seq, blocker, w2_start, a_start, a_done

  def _relocate(self, seq, blocker, a_start, a_done, guard):
    """_try_relocate_blocker_earlier for `blocker`, which pins a_start (latency 1000 us)."""
    positions = {x: i for i, x in enumerate(seq)}
    names = {x.name: i for x, i in positions.items()}
    return pass_lib._try_relocate_blocker_earlier(
        blocker,
        FakeComp("c", seq),
        FakeSchedule(),
        list(seq),
        positions,
        names,
        self.b.comp_by_name,
        a_start,
        positions[a_done],
        1000.0,
        _start_of_done(seq),
        "test",
        None,
        guard,
    )

  def test_total_mode_exempts_path1(self):
    seq, blocker, w2_start, a_start, a_done = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "c")
    # The step alone grows the total deficit (800 -> 840) ...
    self.assertIsNotNone(guard.violation(seq, pass_lib._reinsert(seq, [], 1, blocker)))
    result = self._relocate(seq, blocker, a_start, a_done, guard)
    # ... but path 1 counted the hoist it enables (800 -> 780) and is exempt.
    self.assertIsNotNone(result)
    new_seq = result[0]
    self.assertLess(new_seq.index(blocker), new_seq.index(w2_start))
    self.assertEqual(guard.rejections, 0)

  def test_window_mode_guards_path1(self):
    seq, blocker, _, a_start, a_done = self._build()
    guard = pass_lib._WindowGuard(_start_of_done(seq), self.b.comp_by_name, "c")
    with mock.patch.object(pass_lib, "_WINDOW_GUARD_MODE", "window"):
      self.assertIsNone(self._relocate(seq, blocker, a_start, a_done, guard))
    self.assertEqual(guard.rejections, 1)


if __name__ == "__main__":
  unittest.main()
