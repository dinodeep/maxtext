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

"""Unit tests for the NNX branch of gradient_accumulation_loss_and_grad."""

# pylint: disable=too-many-positional-arguments

import functools
import unittest
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from qwix._src.core import qarray

from maxtext.common.common_types import ShardMode
from maxtext.utils import ga_inplace
from maxtext.utils import gradient_accumulation


@dataclass
class _Cfg:
  gradient_accumulation_steps: int = 2
  use_tunix_gradient_accumulation: bool = False
  shard_optimizer_over_data: bool = False
  shard_mode: int = ShardMode.AUTO
  ici_data_parallelism: int = 1
  debug_sharding: bool = False
  training_objective: str = "causal_lm"
  use_tunix_gradient_accumulation: bool = False


class _TinyNNX(nnx.Module):
  """Single linear layer NNX model."""

  def __init__(self, rngs: nnx.Rngs):
    self.linear = nnx.Linear(2, 1, rngs=rngs)

  def __call__(self, x):
    return self.linear(x)


def _fake_loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """A loss_fn shaped like the production loss_fn but for a tiny linear model.

  Returns (loss, aux) where aux follows the schema gradient_accumulation_loss_and_grad
  reads from: xent_sum / total_weights / moe_lb_loss / indexer_loss / mtp_loss.
  """
  del config, dropout_rng, params, is_train
  pred = model(data["inputs"])
  per_sample_loss = jnp.mean((pred - data["targets"]) ** 2, axis=-1)
  xent_sum = jnp.sum(per_sample_loss)
  total_weights = jnp.array(per_sample_loss.shape[0], dtype=jnp.float32)
  aux = {
      "xent_sum": xent_sum,
      "total_weights": total_weights,
      "moe_lb_loss": jnp.array(0.0),
      "indexer_loss": jnp.array(0.0),
      "mtp_loss": jnp.array(0.0),
  }
  return xent_sum, aux


def _zero_weight_loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """Produces model-dependent sums with a zero token denominator."""
  del config, dropout_rng, params, is_train
  xent_sum = jnp.sum(model(data["inputs"]) ** 2) + 1.0
  aux = {
      "xent_sum": xent_sum,
      "total_weights": jnp.array(0.0),
      "moe_lb_loss": jnp.array(0.0),
      "indexer_loss": jnp.array(0.0),
      "mtp_loss": jnp.array(0.0),
  }
  return xent_sum, aux


def _normalized_loss_fn(model, config, data, dropout_rng, params, is_train=True):
  """Produces the per-microbatch normalized loss used by Tunix accumulation."""
  xent_sum, aux = _fake_loss_fn(model, config, data, dropout_rng, params, is_train=is_train)
  return xent_sum / aux["total_weights"], aux


def _make_cotangent_clip(bound):
  """Identity forward; the backward clips the cotangent to [-bound, bound].

  This stands in for a static (fixed-range) fp8 calibration of backward cotangents, e.g.
  bwd_quantization_calibration_method=fixed,0.01, which saturates values outside the range.
  """

  @jax.custom_vjp
  def clip(x):
    return x

  def fwd(x):
    return x, None

  def bwd(_, g):
    return (jnp.clip(g, -bound, bound),)

  clip.defvjp(fwd, bwd)
  return clip


def _make_fixed_fp8_bwd_probe(seen, quantize=True):
  """Identity forward; the backward records its cotangent, then (if quantize) quantizes it with qwix e5m2 `fixed,0.01`.

  This is the recipe's backward calibration (bwd_quantization_calibration_method=fixed,0.01): a static per-tensor
  range, so a cotangent larger than the range saturates.
  """

  @jax.custom_vjp
  def probe(x):
    return x

  def fwd(x):
    return x, None

  def bwd(_, g):
    jax.debug.callback(lambda v: seen.append(np.asarray(v)), g)
    if not quantize:
      return (g,)
    how = qarray.HowToQuantize(qtype=jnp.float8_e5m2, tiled_axes={}, calibration_method="fixed,0.01")
    return (qarray.dequantize(qarray.quantize(g, how)).astype(g.dtype),)

  probe.defvjp(fwd, bwd)
  return probe


class TestGradientAccumulationNNX(unittest.TestCase):
  """Cover the NNX path of gradient_accumulation_loss_and_grad."""

  def setUp(self):
    self.model = _TinyNNX(rngs=nnx.Rngs(0))
    self.cfg = _Cfg(gradient_accumulation_steps=2)
    # 4 examples → 2 microbatches of 2 each
    self.data = {
        "inputs": jnp.arange(8.0).reshape(4, 2),
        "targets": jnp.zeros((4, 1)),
    }

  def _params_shardings(self):
    """Build a per-leaf NamedSharding tree shaped like nnx.split(model, nnx.Param, ...)[1].

    Uses a trivial single-device mesh so jax.lax.with_sharding_constraint accepts the
    sharding without contradicting the actual device topology.
    """
    _, params, _ = nnx.split(self.model, nnx.Param, ...)
    mesh = Mesh(
        np.array(jax.local_devices()[:1]).reshape(
            1,
        ),
        ("x",),
    )
    ns = NamedSharding(mesh, PartitionSpec())
    return jax.tree.map(lambda _: ns, params)

  def test_nnx_path_runs_and_returns_grad_for_every_param(self):
    """The NNX branch must call nnx.value_and_grad and return one gradient per Param."""
    loss, aux, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,  # NNX branch ignores params
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
    )
    self.assertTrue(jnp.isfinite(loss))
    self.assertIn("xent_sum", aux)
    self.assertIn("total_weights", aux)
    grad_leaves = jax.tree.leaves(raw_grads)
    self.assertEqual(len(grad_leaves), 2)  # linear.kernel + linear.bias
    for g in grad_leaves:
      self.assertTrue(jnp.all(jnp.isfinite(g)))

  def test_positive_weights_match_direct_causal_batch(self):
    """The guarded denominator preserves ordinary causal accumulation."""
    kernel = self.model.linear.kernel.get_value()
    bias = self.model.linear.bias.get_value()

    def direct_loss(kernel, bias):
      predictions = self.data["inputs"] @ kernel + bias
      return jnp.mean((predictions - self.data["targets"]) ** 2)

    expected_loss, expected_grads = jax.value_and_grad(direct_loss, argnums=(0, 1))(kernel, bias)
    loss, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
    )

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(raw_grads["linear"]["kernel"].get_value(), expected_grads[0], rtol=1e-6)
    np.testing.assert_allclose(raw_grads["linear"]["bias"].get_value(), expected_grads[1], rtol=1e-6)

  def test_backward_cotangents_keep_ga1_magnitude(self):
    """A range-limited backward sees the same cotangents under GA as without it.

    The manual GA loss is the token-summed xent. Differentiated as is, its cotangents are W times the GA=1 ones and a
    fixed-range backward calibration saturates them. The accumulation scales the differentiated loss by 1 / (token
    capacity), so the GA=2 gradient equals the GA=1 gradient of the same batch through the same clipped backward.
    """
    self.cfg.global_batch_size_to_train_on = 4
    self.cfg.max_target_length = 1  # token capacity 4 = total_weights of the 4-example batch
    kernel = self.model.linear.kernel.get_value()
    bias = self.model.linear.bias.get_value()
    preds = self.data["inputs"] @ kernel + bias
    ga1_cotangent = jnp.max(jnp.abs(2.0 * (preds - self.data["targets"]) / 4.0))
    bound = float(1.5 * ga1_cotangent)
    # The unscaled xent_sum cotangent (4x the GA=1 one) must exceed the bound, or this test checks nothing.
    self.assertGreater(float(jnp.max(jnp.abs(2.0 * (preds - self.data["targets"])))), bound)
    clip = _make_cotangent_clip(bound)

    def direct_loss(kernel, bias):
      predictions = clip(self.data["inputs"] @ kernel + bias)
      return jnp.mean((predictions - self.data["targets"]) ** 2)

    def clipped_loss_fn(model, config, data, dropout_rng, params, is_train=True):
      del config, dropout_rng, params, is_train
      per_sample_loss = jnp.mean((clip(model(data["inputs"])) - data["targets"]) ** 2, axis=-1)
      xent_sum = jnp.sum(per_sample_loss)
      aux = {
          "xent_sum": xent_sum,
          "total_weights": jnp.array(per_sample_loss.shape[0], dtype=jnp.float32),
          "moe_lb_loss": jnp.array(0.0),
          "indexer_loss": jnp.array(0.0),
          "mtp_loss": jnp.array(0.0),
      }
      return xent_sum, aux

    expected_grads = jax.grad(direct_loss, argnums=(0, 1))(kernel, bias)
    _, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        clipped_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
    )
    np.testing.assert_allclose(raw_grads["linear"]["kernel"].get_value(), expected_grads[0], rtol=1e-6)
    np.testing.assert_allclose(raw_grads["linear"]["bias"].get_value(), expected_grads[1], rtol=1e-6)

  def test_ga2_cotangent_at_fixed_quantizer_matches_total_batch(self):
    """GA=2 feeds a fixed-range fp8 backward quantizer the GA=1 cotangent of the concatenated batch.

    With 2N = 8 sequences of one token, the GA=1 loss is the mean over 8, so the cotangent of each per-token loss is
    1/(2N) = 0.125. Under GA=2 the accumulation scales each microbatch's token-summed loss by 1/S with
    S = global_batch_size_to_train_on (8, which includes the GA factor) x max_target_length (1), so the cotangent
    must be 0.125 as well (it was 1 before the fix); that probe only records. A second probe with the recipe's e5m2
    `fixed,0.01` quantizer sits on the predictions, and the final gradient must equal the GA=1 gradient of the
    concatenated batch.
    """
    self.cfg.global_batch_size_to_train_on = 8
    self.cfg.max_target_length = 1
    inputs = jnp.linspace(-1.0, 1.0, 16, dtype=jnp.float32).reshape(8, 2)
    kernel = self.model.linear.kernel.get_value()
    bias = self.model.linear.bias.get_value()
    preds = inputs @ kernel + bias
    residual = jnp.linspace(-0.03, 0.035, 8, dtype=jnp.float32).reshape(8, 1)
    data = {"inputs": inputs, "targets": preds - residual}
    # d(mean over 8 of r^2)/d pred = 2r/8, at most 0.00875 < 0.01; unscaled (2r, up to 0.07) it would saturate.
    self.assertLess(float(jnp.max(jnp.abs(2.0 * residual / 8.0))), 0.01)
    self.assertGreater(float(jnp.max(jnp.abs(2.0 * residual))), 0.01)

    seen_ga1_tok, seen_ga1_pred, seen_ga2_tok, seen_ga2_pred = [], [], [], []

    def direct_loss(kernel, bias):
      tok_probe = _make_fixed_fp8_bwd_probe(seen_ga1_tok, quantize=False)
      pred_probe = _make_fixed_fp8_bwd_probe(seen_ga1_pred)
      predictions = pred_probe(inputs @ kernel + bias)
      per_token = tok_probe(jnp.mean((predictions - data["targets"]) ** 2, axis=-1))
      return jnp.mean(per_token)

    def probed_loss_fn(model, config, data, dropout_rng, params, is_train=True):
      del config, dropout_rng, params, is_train
      tok_probe = _make_fixed_fp8_bwd_probe(seen_ga2_tok, quantize=False)
      pred_probe = _make_fixed_fp8_bwd_probe(seen_ga2_pred)
      per_token = tok_probe(jnp.mean((pred_probe(model(data["inputs"])) - data["targets"]) ** 2, axis=-1))
      xent_sum = jnp.sum(per_token)
      aux = {
          "xent_sum": xent_sum,
          "total_weights": jnp.array(per_token.shape[0], dtype=jnp.float32),
          "moe_lb_loss": jnp.array(0.0),
          "indexer_loss": jnp.array(0.0),
          "mtp_loss": jnp.array(0.0),
      }
      return xent_sum, aux

    expected_grads = jax.grad(direct_loss, argnums=(0, 1))(kernel, bias)
    _, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        probed_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=data,
        dropout_rng=None,
    )
    jax.effects_barrier()

    # Per-token cotangent: GA=1 on the 8-sequence batch sees 1/8 on every token; GA=2 sees the same, twice.
    self.assertEqual(len(seen_ga1_tok), 1)
    self.assertEqual(len(seen_ga2_tok), 2)
    np.testing.assert_array_equal(seen_ga1_tok[0], np.full((8,), 0.125, np.float32))
    for tok in seen_ga2_tok:
      np.testing.assert_array_equal(tok, np.full((4,), 0.125, np.float32))
    # Cotangent at the fp8 quantizer on the predictions. The accumulation splits the batch with a stride of GA
    # (microbatch k holds rows k, k + GA, ...), so microbatch k must see rows k::2 of the GA=1 cotangent.
    self.assertEqual(len(seen_ga2_pred), 2)
    for k, pred_cot in enumerate(seen_ga2_pred):
      np.testing.assert_allclose(pred_cot, seen_ga1_pred[0][k::2], rtol=1e-6, atol=0)
    self.assertLess(float(np.max(np.abs(seen_ga1_pred[0]))), 0.01)
    np.testing.assert_allclose(raw_grads["linear"]["kernel"].get_value(), expected_grads[0], rtol=1e-6)
    np.testing.assert_allclose(raw_grads["linear"]["bias"].get_value(), expected_grads[1], rtol=1e-6)

  def test_tunix_block_diffusion_uses_accumulation_step_divisor(self):
    self.cfg.training_objective = "block_diffusion"
    self.cfg.use_tunix_gradient_accumulation = True
    kernel = self.model.linear.kernel.get_value()
    bias = self.model.linear.bias.get_value()

    def direct_loss(kernel, bias):
      predictions = self.data["inputs"] @ kernel + bias
      return jnp.mean((predictions - self.data["targets"]) ** 2)

    expected_loss, expected_grads = jax.value_and_grad(direct_loss, argnums=(0, 1))(kernel, bias)
    loss, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _normalized_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
    )

    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    np.testing.assert_allclose(raw_grads["linear"]["kernel"].get_value(), expected_grads[0], rtol=1e-6)
    np.testing.assert_allclose(raw_grads["linear"]["bias"].get_value(), expected_grads[1], rtol=1e-6)

  def test_nnx_path_updates_model_rest_state_after_scan(self):
    """After accumulation, nnx.update is called on the model with the rest_state from the scan.

    For a TinyNNX (no rngs/dropout), the rest tree is empty but the call path must still
    succeed end-to-end without raising — covering the `if is_nnx: nnx.update(...)` branch.
    """
    pre_kernel = self.model.linear.kernel.value.copy()
    gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
    )
    # The kernel itself is a Param — gradient_accumulation_loss_and_grad does not apply
    # gradients to params, so the value should be untouched.
    self.assertTrue(jnp.allclose(self.model.linear.kernel.value, pre_kernel))

  def test_nnx_with_shard_optimizer_over_data_casts_to_bf16(self):
    """Zero-1 path must convert fp32 params to bf16 before the scan loop."""
    self.cfg.shard_optimizer_over_data = True
    # Should not raise; just verify the function runs and returns sensible outputs.
    loss, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        self.model,
        params=None,
        params_shardings=self._params_shardings(),
        data=self.data,
        dropout_rng=None,
    )
    self.assertTrue(jnp.isfinite(loss))
    for g in jax.tree.leaves(raw_grads):
      self.assertTrue(jnp.all(jnp.isfinite(g)))

  def test_zero_total_weights_returns_zero_loss_and_gradients(self):
    for training_objective in ("causal_lm", "block_diffusion"):
      for use_tunix_gradient_accumulation in (False, True):
        with self.subTest(
            training_objective=training_objective,
            use_tunix_gradient_accumulation=use_tunix_gradient_accumulation,
        ):
          self.cfg.training_objective = training_objective
          self.cfg.use_tunix_gradient_accumulation = use_tunix_gradient_accumulation
          loss, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
              _zero_weight_loss_fn,
              self.cfg,
              self.model,
              params=None,
              params_shardings=self._params_shardings(),
              data=self.data,
              dropout_rng=None,
          )

          self.assertEqual(float(loss), 0.0)
          for gradient in jax.tree.leaves(raw_grads):
            self.assertTrue(jnp.all(jnp.isfinite(gradient)))
            self.assertTrue(jnp.all(gradient == 0))


class _ScannedStack(nnx.Module):
  """A stack of tanh(h @ kernel + bias) layers, stored like `_create_scanned_layers` stores one."""

  def __init__(self, num_layers, dim, param_scan_axis, rngs):
    metadata = {nnx.PARTITION_NAME: "layers", "param_scan_axis": param_scan_axis}
    kernel = 0.3 * jax.random.normal(rngs.params(), (num_layers, dim, dim))
    bias = 0.1 * jax.random.normal(rngs.params(), (num_layers, dim))
    self.kernel = nnx.Param(jnp.moveaxis(kernel, 0, param_scan_axis), **metadata)
    self.bias = nnx.Param(jnp.moveaxis(bias, 0, param_scan_axis), **metadata)


class _ScannedNNX(nnx.Module):
  """Applies `decoder.layers` with a layer scan that, like NNXDecoder's, accumulates in place."""

  def __init__(self, param_scan_axis, inplace_scan=True, num_layers=3, dim=2, rngs=None):
    self.decoder = nnx.Module()
    self.decoder.layers = _ScannedStack(num_layers, dim, param_scan_axis, rngs)
    self.param_scan_axis = param_scan_axis
    self.inplace_scan = inplace_scan

  def __call__(self, x):
    axis = self.param_scan_axis
    params = nnx.state(self.decoder.layers, nnx.Param)
    params = jax.tree.map(lambda p: jnp.moveaxis(p, axis, 0) if p.ndim > axis else p, params)

    def layer(h, layer_params):
      return jnp.tanh(h @ layer_params["kernel"].get_value() + layer_params["bias"].get_value())

    name = ga_inplace.stack_name(params)
    acc = ga_inplace.lookup(name) if self.inplace_scan else None
    if acc is None or not ga_inplace.stacks(params, acc, axis):
      h, _ = jax.lax.scan(lambda h, p: (layer(h, p), None), x, params)
      return h

    def ga_layer(carry, xs):
      h, a = carry
      a = jax.lax.optimization_barrier(a)
      layer_params, i = xs
      layer_params = ga_inplace.inject_layer_params(layer_params, a, i, axis)
      return (layer(h, layer_params), a), None

    num_layers = jax.tree.leaves(params)[0].shape[0]
    (h, acc_final), _ = jax.lax.scan(ga_layer, (x, acc), (params, jnp.arange(num_layers)))
    ga_inplace.set_final(name, acc_final)
    return h


class TestInPlaceGradientAccumulation(unittest.TestCase):
  """Scanned stacks accumulate their gradients in place, at any param_scan_axis."""

  def setUp(self):
    self.cfg = _Cfg(gradient_accumulation_steps=2)
    self.data = {
        "inputs": jnp.arange(8.0).reshape(4, 2) / 8.0,
        "targets": jnp.ones((4, 2)),
    }

  def _ga_and_direct(self, model):
    """Returns the GA loss and gradients, and those of the whole batch at once."""
    _, params, _ = nnx.split(model, nnx.Param, ...)
    ns = NamedSharding(Mesh(np.array(jax.local_devices()[:1]), ("x",)), PartitionSpec())
    loss, _, raw_grads = gradient_accumulation.gradient_accumulation_loss_and_grad(
        _fake_loss_fn,
        self.cfg,
        model,
        params=None,
        params_shardings=jax.tree.map(lambda _: ns, params),
        data=self.data,
        dropout_rng=None,
    )

    def direct_loss(model):
      return jnp.mean(jnp.mean((model(self.data["inputs"]) - self.data["targets"]) ** 2, axis=-1))

    expected_loss, expected_grads = nnx.value_and_grad(direct_loss)(model)
    return loss, raw_grads, expected_loss, expected_grads

  def _assert_matches_full_batch(self, model):
    loss, raw_grads, expected_loss, expected_grads = self._ga_and_direct(model)
    np.testing.assert_allclose(loss, expected_loss, rtol=1e-6)
    for name in ("kernel", "bias"):
      np.testing.assert_allclose(
          raw_grads["decoder"]["layers"][name].get_value(),
          expected_grads["decoder"]["layers"][name].get_value(),
          rtol=1e-5,
          atol=1e-7,
      )

  def test_matches_full_batch_at_any_param_scan_axis(self):
    for param_scan_axis in (0, 1):
      with self.subTest(param_scan_axis=param_scan_axis):
        self._assert_matches_full_batch(_ScannedNNX(param_scan_axis, rngs=nnx.Rngs(0)))

  def test_stack_whose_scan_does_not_pick_up_its_accumulator(self):
    """A stack applied by a scan that does not accumulate in place accumulates as usual."""
    self._assert_matches_full_batch(_ScannedNNX(0, inplace_scan=False, rngs=nnx.Rngs(0)))

  def test_stack_paths_and_names(self):
    """A stack is found by the scan name its parameters carry."""
    model = _ScannedNNX(1, rngs=nnx.Rngs(0))
    _, params, _ = nnx.split(model, nnx.Param, ...)
    self.assertEqual(ga_inplace.stack_paths(params), {"layers": ("decoder", "layers")})
    self.assertEqual(ga_inplace.stack_name(nnx.state(model.decoder.layers, nnx.Param)), "layers")
    self.assertIsNone(ga_inplace.stack_name(nnx.state(_TinyNNX(nnx.Rngs(0)), nnx.Param)))

  def test_inject_routes_the_layer_cotangent_along_the_scan_axis(self):
    for param_scan_axis in (0, 1):
      with self.subTest(param_scan_axis=param_scan_axis):
        acc = jnp.moveaxis(jnp.arange(24.0).reshape(3, 2, 4), 0, param_scan_axis)
        w = jnp.ones((2, 4))
        inject = functools.partial(ga_inplace.inject_layer_params, i=1, param_scan_axis=param_scan_axis)
        _, vjp = jax.vjp(lambda w_, a, inject=inject: inject(w_, a), w, acc)
        ct_w, ct_acc = vjp(jnp.full((2, 4), 5.0))
        np.testing.assert_array_equal(ct_w, jnp.zeros((2, 4)))
        expected = jnp.zeros((3, 2, 4)).at[1].set(5.0)
        np.testing.assert_array_equal(ct_acc, jnp.moveaxis(expected, 0, param_scan_axis))


if __name__ == "__main__":
  unittest.main()
