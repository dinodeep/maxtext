# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for optional TransformerEngine MoE arguments."""

# These tests directly exercise the private checkpoint compatibility helper.
# pylint: disable=protected-access

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from maxtext.layers import moe


def test_te_moe_checkpoint_kwargs_for_supported_signature():
  def te_moe(
      *,
      wi_0_checkpoint_name=None,
      wi_1_checkpoint_name=None,
      wo_checkpoint_name=None,
      dispatch_checkpoint_name=None,
      combine_checkpoint_name=None,
  ):
    del wi_0_checkpoint_name, wi_1_checkpoint_name, wo_checkpoint_name
    del dispatch_checkpoint_name, combine_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {
      "wi_0_checkpoint_name": "moe_mlpwi_0",
      "wi_1_checkpoint_name": "moe_mlpwi_1",
      "wo_checkpoint_name": "moe_mlpwo",
      "dispatch_checkpoint_name": "te_moe_dispatch_remat_location",
  }


def test_te_moe_checkpoint_kwargs_for_previous_signature():
  def te_moe(*, wi_0_checkpoint_name=None, wi_1_checkpoint_name=None, wo_checkpoint_name=None):
    del wi_0_checkpoint_name, wi_1_checkpoint_name, wo_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {
      "wi_0_checkpoint_name": "moe_mlpwi_0",
      "wi_1_checkpoint_name": "moe_mlpwi_1",
      "wo_checkpoint_name": "moe_mlpwo",
  }


def test_te_moe_checkpoint_kwargs_for_partial_signature():
  def older_te_moe(*, wi_0_checkpoint_name=None, wi_1_checkpoint_name=None):
    del wi_0_checkpoint_name, wi_1_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(older_te_moe) == {
      "wi_0_checkpoint_name": "moe_mlpwi_0",
      "wi_1_checkpoint_name": "moe_mlpwi_1",
  }


def test_te_moe_checkpoint_kwargs_for_ep_only_signature():
  def te_moe(*, dispatch_checkpoint_name=None):
    del dispatch_checkpoint_name

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {
      "dispatch_checkpoint_name": "te_moe_dispatch_remat_location",
  }


def test_te_moe_checkpoint_kwargs_for_legacy_signature():
  def te_moe():
    pass

  assert moe._get_te_moe_checkpoint_kwargs(te_moe) == {}


def test_te_moe_cudnn_gmm_fusion_default_is_omitted():
  assert not moe._get_te_moe_cudnn_gmm_fusion_kwargs(False)


def test_te_moe_cudnn_gmm_fusion_can_be_disabled():
  assert moe._get_te_moe_cudnn_gmm_fusion_kwargs(True) == {"use_cudnn_fusion": False}


def test_te_moe_weight_layout_is_standard_when_supported():
  def te_moe(*, cudnn_native_weight_layout=None):
    del cudnn_native_weight_layout

  assert moe._get_te_moe_weight_layout_kwargs(te_moe) == {"cudnn_native_weight_layout": False}


def test_te_moe_weight_layout_is_omitted_for_older_signature():
  def te_moe():
    pass

  assert not moe._get_te_moe_weight_layout_kwargs(te_moe)


def test_te_moe_native_weight_layout_kwargs():
  def te_moe(*, cudnn_native_weight_layout=None, fc2_native_weight_layout=False):
    del cudnn_native_weight_layout, fc2_native_weight_layout

  assert moe._get_te_moe_weight_layout_kwargs(te_moe, native_weight_layout=True) == {
      "cudnn_native_weight_layout": True,
      "fc2_native_weight_layout": True,
  }


def test_te_moe_overlap_kwargs():
  def te_moe(*, enable_ep_overlap=False):
    del enable_ep_overlap

  assert not moe._get_te_moe_overlap_kwargs(te_moe, False)
  assert moe._get_te_moe_overlap_kwargs(te_moe, True) == {"enable_ep_overlap": True}


def test_te_moe_fc1_to_native_preserves_values():
  pytest.importorskip("transformer_engine.jax.cpp_extensions")
  wi = jax.random.normal(jax.random.key(0), (2, 4, 128))  # [E, K, 2N], N = 64: two 32-row blocks per half
  native = moe.te_moe_fc1_to_native(wi)
  assert native.shape == (2, 128, 4)
  gate, up = wi[..., :64], wi[..., 64:]
  np.testing.assert_array_equal(native[:, 0:32], jnp.swapaxes(gate[..., 0:32], -1, -2))
  np.testing.assert_array_equal(native[:, 32:64], jnp.swapaxes(up[..., 0:32], -1, -2))
  np.testing.assert_array_equal(native[:, 64:96], jnp.swapaxes(gate[..., 32:64], -1, -2))
  np.testing.assert_array_equal(native[:, 96:128], jnp.swapaxes(up[..., 32:64], -1, -2))
