# Copyright 2025-2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""In-place gradient accumulation for scanned layer stacks.

With plain gradient accumulation, each microbatch's backward pass writes the gradients of a
scanned layer stack into a fresh `[num_layers, ...]` buffer, which is then added to the running
accumulator. Both buffers are live for the whole backward pass, so GA costs one extra copy of
the stacked parameters' gradients.

Here the accumulator is instead threaded through the layer scan as a carry, and each layer's
parameter slice goes through `_inject`, which routes the slice's cotangent into the carried
accumulator at that layer's index. A dead linear term `<stop_gradient(acc), acc_final>` in the
loss seeds the backward carry cotangent with the running accumulator, so the backward scan adds
every layer's gradient into it in place (XLA turns `acc + dynamic_update_slice(0, g, i)` into an
in-place update), and the gradient of the loss w.r.t. the accumulator *is* the new accumulator.

The accumulator keeps the stored parameter layout, with the layer axis at `param_scan_axis`, so
that it never needs a transposed copy; each layer's cotangent is written along that axis.

A stack is identified by its scan name, the `nnx.PARTITION_NAME` that `_create_scanned_layers`
records on each of its parameters. A stack whose scan does not pick up its accumulator (e.g. one
applied by another code path) keeps the regular accumulation.

Usage (see gradient_accumulation.py):
  paths = stack_paths(params)                 # scan name -> path of the stack in the gradient tree
  with accumulate_into({name: grad[path] for name, path in paths.items()}) as finals:
    loss, aux = loss_fn(...)                  # NNXDecoder's layer scan picks up its stack's accumulator
  for name, acc_final in finals.items():      # only the scans that picked up their accumulator
    loss += linear_seed_term(accs[name], acc_final)
"""

import contextlib
import functools

from flax import nnx
import jax
import jax.numpy as jnp

# Name (the scan's metadata_axis_name) -> accumulator tree, active while tracing a GA microbatch.
_ACCUMULATORS = {}
# Name -> accumulator tree as it leaves the layer scan.
_FINALS = {}


@contextlib.contextmanager
def accumulate_into(accumulators):
  """Registers accumulator trees for the layer scans named in `accumulators` while tracing."""
  if _ACCUMULATORS:
    raise RuntimeError("ga_inplace.accumulate_into does not nest.")
  _ACCUMULATORS.update(accumulators)
  _FINALS.clear()
  try:
    yield _FINALS
  finally:
    _ACCUMULATORS.clear()


def lookup(name):
  """Returns the accumulator tree registered for the layer scan `name`, or None."""
  return _ACCUMULATORS.get(name)


def _scan_name(variable):
  return variable.get_metadata().get(nnx.PARTITION_NAME) if isinstance(variable, nnx.Variable) else None


def stack_name(stacked_params):
  """Returns the scan name all parameters of a scanned stack carry, or None if they do not share one."""
  names = {_scan_name(v) for v in jax.tree.leaves(stacked_params, is_leaf=lambda x: isinstance(x, nnx.Variable))}
  return names.pop() if len(names) == 1 else None


def stack_paths(params):
  """Maps the scan name of every scanned stack in the parameter state `params` to the stack's path.

  The stack is the deepest module holding every parameter with that scan name.
  """
  paths_by_name = {}
  for path, variable in nnx.to_flat_state(params):
    name = _scan_name(variable)
    if name is not None:
      paths_by_name.setdefault(name, []).append(tuple(path))
  roots = {}
  for name, paths in paths_by_name.items():
    prefix = paths[0][:-1]
    for path in paths[1:]:
      common = 0
      while common < min(len(prefix), len(path)) and prefix[common] == path[common]:
        common += 1
      prefix = prefix[:common]
    if prefix:
      roots[name] = prefix
  return roots


def stacks(stacked_params, acc, param_scan_axis):
  """Whether `acc` has one leaf per leaf of `stacked_params` (layer axis leading), in the stored layout."""
  p_leaves, a_leaves = jax.tree.leaves(stacked_params), jax.tree.leaves(acc)
  if len(p_leaves) != len(a_leaves):
    return False
  for p, a in zip(p_leaves, a_leaves):
    axis = param_scan_axis if a.ndim > param_scan_axis else 0
    if a.shape != p.shape[1 : axis + 1] + p.shape[:1] + p.shape[axis + 1 :]:
      return False
  return True


def set_final(name, acc_final):
  _FINALS[name] = acc_final


@functools.partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2))
def _inject(shape, dtype, axis, w, acc, i):
  """Identity on the layer slice `w`; its cotangent is routed into `acc[i]` along `axis` instead."""
  del shape, dtype, axis, acc, i
  return w


def _inject_fwd(shape, dtype, axis, w, acc, i):
  del shape, dtype, axis, acc
  return w, i


def _inject_bwd(shape, dtype, axis, i, ct):
  ct_acc = jax.lax.dynamic_update_index_in_dim(jnp.zeros(shape, dtype), ct.astype(dtype), i, axis)
  return None, ct_acc, None


_inject.defvjp(_inject_fwd, _inject_bwd)


def inject_layer_params(layer_params, acc, i, param_scan_axis):
  """Applies `_inject` to every leaf of one layer's parameter slice against the stacked `acc`.

  `acc` has the stored layout: its layer axis is `param_scan_axis`, or 0 for a stack too low-rank
  to hold that axis, as the decoder moves it for the scan.
  """
  p_leaves, treedef = jax.tree.flatten(layer_params)
  a_leaves = jax.tree.leaves(acc)
  if len(p_leaves) != len(a_leaves):
    raise ValueError(f"accumulator has {len(a_leaves)} leaves, layer params have {len(p_leaves)}.")
  out = []
  for p, a in zip(p_leaves, a_leaves):
    axis = param_scan_axis if a.ndim > param_scan_axis else 0
    if a.shape[:axis] + a.shape[axis + 1 :] != p.shape:
      raise ValueError(f"accumulator leaf {a.shape} does not stack layer param leaf {p.shape} on axis {axis}.")
    out.append(_inject(a.shape, a.dtype, axis, p, a, i))
  return jax.tree.unflatten(treedef, out)


def linear_seed_term(acc, acc_final):
  """sum(stop_gradient(acc) * acc_final): dead in the forward pass, seeds the backward with acc."""
  terms = [
      jnp.sum(jax.lax.stop_gradient(a).astype(jnp.float32) * f.astype(jnp.float32))
      for a, f in zip(jax.tree.leaves(acc), jax.tree.leaves(acc_final))
  ]
  return sum(terms)
