"""Tests for the te_precompute_weight_amax helpers (no TE or GPU needed)."""

import unittest

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from maxtext.layers import quantizations
from maxtext.utils import maxtext_utils_nnx


class _Dense(nnx.Module):
  """Minimal stand-in for DenseGeneral: a kernel Param, a compute dtype and a TEKernelAmax."""

  def __init__(self, kernel, amax, dtype=jnp.bfloat16, scan_axis=None):
    meta = {} if scan_axis is None else {"param_scan_axis": scan_axis}
    self.kernel = nnx.Param(kernel, **meta)
    self.kernel_amax = None if amax is None else quantizations.TEKernelAmax(amax)
    self.dtype = dtype


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


if __name__ == "__main__":
  unittest.main()
