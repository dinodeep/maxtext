# Copyright 2026 Google LLC
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

"""Tests for the te_precompute_weight_amax helpers (no TE or GPU needed)."""

import sys
import types
import unittest
from unittest import mock

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils_nnx


class _Dense(nnx.Module):
  """Minimal stand-in for DenseGeneral: a kernel Param, a compute dtype and a TEKernelAmax."""

  def __init__(self, kernel, amax, dtype=jnp.bfloat16, scan_axis=None, *, parameter_memory_host_offload=False):
    meta = {} if scan_axis is None else {"param_scan_axis": scan_axis}
    self.kernel = nnx.Param(kernel, **meta)
    self.kernel_amax = None if amax is None else quantizations.TEKernelAmax(amax)
    self.dtype = dtype
    self.parameter_memory_host_offload = parameter_memory_host_offload


class _Model(nnx.Module):

  def __init__(self, *layers):
    self.layers = nnx.List(layers) if hasattr(nnx, "List") else list(layers)


class TEKernelAmaxTest(unittest.TestCase):

  def test_unstacked_kernel(self):
    k = jnp.array([[0.5, -3.0], [2.0, 1.0]], jnp.float32)
    m = _Model(_Dense(k, jnp.zeros((1,), jnp.float32)))
    self.assertEqual(quantizations.refresh_te_kernel_amaxes(m), 1)
    np.testing.assert_array_equal(m.layers[0].kernel_amax.get_value(), np.array([3.0], np.float32))

  def test_stacked_kernel_uses_param_scan_axis(self):
    # [embed, layers, out] with the layer axis at 1, as with param_scan_axis=1.
    k = jnp.stack([jnp.full((4, 3), v, jnp.float32) for v in (1.0, -5.0, 2.5)], axis=1)
    k = k.at[0, 2, 0].set(-7.0)
    m = _Model(_Dense(k, jnp.zeros((3, 1), jnp.float32), scan_axis=1))
    quantizations.refresh_te_kernel_amaxes(m)
    np.testing.assert_array_equal(m.layers[0].kernel_amax.get_value(), np.array([[1.0], [5.0], [7.0]], np.float32))

  def test_amax_is_taken_in_compute_dtype(self):
    k = jnp.array([1.0 + 2.0**-12], jnp.float32)  # rounds to 1.0 in bf16
    m = _Model(_Dense(k, jnp.zeros((1,), jnp.float32), dtype=jnp.bfloat16))
    quantizations.refresh_te_kernel_amaxes(m)
    self.assertEqual(float(m.layers[0].kernel_amax.get_value()[0]), 1.0)

  def test_amax_is_rounded_to_compute_dtype_under_jit(self):
    # 0.2009 rounds up to 0.201171875 in bf16. Under jit, XLA may drop a float32 -> bf16 -> float32 cast round trip,
    # which would leave the amax below the largest bf16 value that TE quantizes.
    # The kernel must be large enough that XLA does not constant-fold or special-case it.
    k = jnp.zeros((128, 256), jnp.float32).at[3, 5].set(0.2009).at[7, 9].set(-0.1)
    amax = jax.jit(quantizations.kernel_amax_like, static_argnums=1)(k, jnp.bfloat16)
    self.assertEqual(float(amax), float(jnp.max(jnp.abs(k.astype(jnp.bfloat16))).astype(jnp.float32)))
    self.assertEqual(float(amax), 0.201171875)

  def test_mismatched_stack_raises(self):
    k = jnp.ones((4, 3, 2), jnp.float32)
    m = _Model(_Dense(k, jnp.zeros((5, 1), jnp.float32), scan_axis=1))
    with self.assertRaisesRegex(ValueError, "does not match"):
      quantizations.refresh_te_kernel_amaxes(m)

  def test_modules_without_amax_are_untouched(self):
    m = _Model(_Dense(jnp.ones((2, 2)), None))
    self.assertEqual(quantizations.refresh_te_kernel_amaxes(m), 0)

  def test_refresh_under_jit_is_not_differentiated(self):
    k = jnp.array([[1.0, -4.0]], jnp.float32)
    m = _Model(_Dense(k, jnp.zeros((1,), jnp.float32)))
    graphdef, state = nnx.split(m)

    def f(state):
      model = nnx.merge(graphdef, state)
      quantizations.refresh_te_kernel_amaxes(model)
      return model.layers[0].kernel_amax.get_value()

    self.assertEqual(float(jax.jit(f)(state)[0]), 4.0)

  def _refresh_under_jit(self, model):
    graphdef, state = nnx.split(model)

    def f(state):
      m = nnx.merge(graphdef, state)
      quantizations.refresh_te_kernel_amaxes(m)
      return [layer.kernel_amax.get_value() for layer in m.layers]

    return jax.jit(f)(state)

  def test_host_offloaded_kernels_match_device_kernels(self):
    # Stacked kernel [layers, embed, out] (param_scan_axis=0, as host offload requires) and an unstacked kernel.
    stacked = jax.random.normal(jax.random.PRNGKey(0), (3, 8, 4), jnp.float32) * jnp.array([1.0, 5.0, 0.5])[:, None, None]
    single = jax.random.normal(jax.random.PRNGKey(1), (8, 4), jnp.float32)

    def model(offload):
      return _Model(
          _Dense(stacked, jnp.zeros((3, 1), jnp.float32), scan_axis=0, parameter_memory_host_offload=offload),
          _Dense(single, jnp.zeros((1,), jnp.float32), parameter_memory_host_offload=offload),
      )

    expected = self._refresh_under_jit(model(False))
    got = self._refresh_under_jit(model(True))
    for e, g in zip(expected, got):
      np.testing.assert_array_equal(g, e)

  def test_host_offloaded_kernel_in_pinned_host_memory(self):
    device = jax.devices()[0]
    if "pinned_host" not in [m.kind for m in device.addressable_memories()]:
      self.skipTest("backend has no pinned_host memory")
    stacked = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) - 20.0
    host = jax.sharding.SingleDeviceSharding(device, memory_kind="pinned_host")
    m = _Model(
        _Dense(
            jax.device_put(stacked, host), jnp.zeros((2, 1), jnp.float32), scan_axis=0, parameter_memory_host_offload=True
        )
    )
    (amax,) = self._refresh_under_jit(m)
    np.testing.assert_array_equal(amax, np.array([[20.0], [8.0]], np.float32))

  def test_host_offload_requires_scan_axis_0(self):
    k = jnp.ones((4, 3, 2), jnp.float32)
    m = _Model(_Dense(k, jnp.zeros((3, 1), jnp.float32), scan_axis=1, parameter_memory_host_offload=True))
    with self.assertRaisesRegex(ValueError, "param_scan_axis=0"):
      quantizations.refresh_te_kernel_amaxes(m)


class TEKernelAmaxScanMetadataTest(unittest.TestCase):

  def test_scan_axis_round_trip_keeps_metadata(self):
    """The layer scans strip and restore the scan axis; a stacked amax must come back unchanged (it is a carry)."""
    stacked = quantizations.TEKernelAmax(
        jnp.ones((3, 1), jnp.float32), out_sharding=("moe_layers", None), partition_name="moe_layers", param_scan_axis=0
    )
    sliced = maxtext_utils_nnx.nnx_remove_scan_axis(stacked.replace(value=stacked.get_value()[0]))
    restored = maxtext_utils_nnx.nnx_add_and_sync_scan_axis(sliced.replace(value=stacked.get_value()))
    self.assertEqual(restored.get_metadata()["out_sharding"], stacked.get_metadata()["out_sharding"])
    self.assertEqual(jax.tree.structure(restored), jax.tree.structure(stacked))


def _fake_te_with_dense(dense):
  """sys.modules entries for a fake transformer_engine whose jax.dense.dense is `dense`."""
  te = types.ModuleType("transformer_engine")
  te_jax = types.ModuleType("transformer_engine.jax")
  te_dense = types.ModuleType("transformer_engine.jax.dense")
  te_dense.dense = dense
  te_jax.dense = te_dense
  te.jax = te_jax
  return {"transformer_engine": te, "transformer_engine.jax": te_jax, "transformer_engine.jax.dense": te_dense}


class TEKernelAmaxVersionCheckTest(unittest.TestCase):

  def test_te_without_kernel_amax_is_rejected(self):
    def dense(x, kernel, contracting_dims=None, quantizer_set=None):
      del x, kernel, contracting_dims, quantizer_set

    with mock.patch.dict(sys.modules, _fake_te_with_dense(dense)):
      with self.assertRaisesRegex(ValueError, "te_precompute_weight_amax=True requires a TransformerEngine"):
        quantizations._check_te_dense_accepts_kernel_amax()  # pylint: disable=protected-access

  def test_te_with_kernel_amax_is_accepted(self):
    def dense(x, kernel, contracting_dims=None, quantizer_set=None, kernel_amax=None):
      del x, kernel, contracting_dims, quantizer_set, kernel_amax

    with mock.patch.dict(sys.modules, _fake_te_with_dense(dense)):
      quantizations._check_te_dense_accepts_kernel_amax()  # pylint: disable=protected-access


if __name__ == "__main__":
  unittest.main()
