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

"""Tests that gpu_collective_overlap_pass keys te_ep (EP) async windows by the
custom-call they wrap, never moves a chased start above its operand, and caches
its cost lookups.

Uses small fake HLO objects (the pass only needs name/opcode/operands/users/
to_string and the computation's instructions)."""

# pylint: disable=protected-access

import unittest
from unittest import mock

import pytest

from maxtext.utils import gpu_collective_overlap_pass as pass_lib

# Pure-Python HLO rewriting; needs no accelerator.
pytestmark = pytest.mark.cpu_only

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


class _PassTestCase(unittest.TestCase):
  """Isolates the module-level cost state of the pass per test."""

  def setUp(self):
    super().setUp()
    self.enterContext(mock.patch.object(pass_lib, "_fingerprinted_costs", {}))
    self.enterContext(mock.patch.object(pass_lib, "_applied_fingerprint_names", set()))
    self.enterContext(mock.patch.object(pass_lib, "_profile_costs", {}))
    self.enterContext(mock.patch.object(pass_lib, "_inst_cost_cache", {}))
    self.enterContext(mock.patch.object(pass_lib, "_te_ep_overrides", {}))
    self.enterContext(mock.patch.object(pass_lib, "_host_offload_copy_names", set()))
    self.enterContext(mock.patch.object(pass_lib, "_wrapped_custom_call_cache", {}))
    self.b = _Builder()

  def select_module(self, fingerprint: str):
    """Apply the pooled costs of the module with `fingerprint`, as the pass does
    for a module it schedules (the fingerprint is read from its HLO proto)."""
    with mock.patch.object(pass_lib, "_module_fingerprint", return_value=fingerprint):
      pass_lib._select_module_profile(b"")

  def load_costs(self):
    """Pool the builder's costs as fingerprinted PGLE costs and apply them."""
    _, by_fingerprint = pass_lib._split_fingerprinted_costs(
        {f"{_FP}::{name}": cost for name, cost in self.b.costs.items()}
    )
    for fingerprint, costs in by_fingerprint.items():
      pass_lib._fingerprinted_costs.setdefault(fingerprint, {}).update(costs)
    self.select_module(_FP)


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
    self.select_module("another_module_fingerprint")
    self.assertEqual(pass_lib._wrapped_custom_call_cache, {})

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
    # No custom-call to key by: the key the lookup found before, the start's own name.
    self.assertEqual(pass_lib._resolve_profile_key(start, self.b.comp_by_name), "async-start.5")


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


if __name__ == "__main__":
  unittest.main()
