# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MaxText shape adapters for Transformer Engine DeepSeek-V4 CSA APIs.

Transformer Engine owns the per-kernel VJPs; its implementations call the raw
cuDNN Frontend CuTeDSL bindings. Imports are deliberately local so ordinary
MaxText configurations do not require an SM100-capable stack.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp


# CSA is independent between sequences.  Treat every outer data-parallel axis
# as a batch partition inside TE's shard_map, including MoE EP and FSDP.
_CSA_BATCH_AXES = ("data", "fsdp", "expert")


def compress_ratio4(kv: Any, gate: Any, position_bias: Any) -> Any:
  """Runs the differentiable ratio-4 overlap compressor, before RMSNorm/RoPE."""
  from transformer_engine.jax.deepseek_v4 import csa_compressor_batched  # pylint: disable=import-outside-toplevel

  batch, sequence, width = kv.shape
  if sequence % 4:
    raise ValueError("cuDNN DSv4 CSA requires sequence length divisible by 4")
  if gate.shape != kv.shape or position_bias.shape != (4, width):
    raise ValueError(f"expected gate={kv.shape} and position_bias={(4, width)}")
  del batch, width
  out = csa_compressor_batched(
      kv.astype(jnp.bfloat16),
      gate.astype(jnp.bfloat16),
      position_bias.astype(jnp.float32),
      batch_axes=_CSA_BATCH_AXES,
  )
  return out


def indexer_ratio4(q: Any, compressed_k: Any, weights: Any, *, softmax_scale: float) -> Any:
  """Dense ratio-4 indexer scores through Transformer Engine's VJP API."""
  from transformer_engine.jax.deepseek_v4 import dsa_indexer_batched  # pylint: disable=import-outside-toplevel

  q_bshd = jnp.transpose(q, (0, 2, 1, 3))
  return dsa_indexer_batched(
      q_bshd.astype(jnp.bfloat16),
      compressed_k[:, :, None, :].astype(jnp.bfloat16),
      weights.astype(jnp.bfloat16),
      batch_axes=_CSA_BATCH_AXES,
      ratio=4,
      sm_scale=softmax_scale,
  )


def sparse_attention_ratio4(
    q: Any,
    local_kv: Any,
    compressed_kv: Any,
    compressed_mask: Any,
    sinks: Any,
    *,
    indexer_topk: int = 512,
    window_size: int = 128,
) -> Any:
  """Runs combined 512 compressed + 128 local-token sparse attention."""
  from transformer_engine.jax.deepseek_v4 import dsa_sparse_attention_batched  # pylint: disable=import-outside-toplevel

  batch, q_len, heads, head_dim = q.shape
  if (heads, head_dim, indexer_topk, window_size) != (64, 512, 512, 128):
    raise ValueError("cuDNN DSv4 sparse attention requires H=64, D=512, top-k=512, window=128")
  if local_kv.shape != (batch, q_len, 1, head_dim):
    raise ValueError("cuDNN DSv4 training requires self-attention with one local KV head")
  comp_len = compressed_kv.shape[1]
  mask = compressed_mask[:, 0] if compressed_mask.ndim == 4 else compressed_mask
  valid = mask > -1.0e20
  rank_scores = jnp.where(valid, jnp.zeros(mask.shape, jnp.float32), -jnp.inf)
  if comp_len < indexer_topk:
    rank_scores = jnp.pad(rank_scores, ((0, 0), (0, 0), (0, indexer_topk - comp_len)), constant_values=-jnp.inf)
  _, selected = jax.lax.top_k(rank_scores, indexer_topk)
  selected_valid = jnp.take_along_axis(rank_scores, selected, axis=-1) > -jnp.inf

  kv = jnp.concatenate([local_kv, compressed_kv], axis=1)[:, :, 0, :]
  compressed_indices = q_len + selected
  compressed_indices = jnp.where(selected_valid & (selected < comp_len), compressed_indices, -1)

  query_pos = jnp.arange(q_len, dtype=jnp.int32)[None, :, None]
  local_pos = query_pos + jnp.arange(1 - window_size, 1, dtype=jnp.int32)[None, None, :]
  local_indices = jnp.broadcast_to(local_pos, (batch, q_len, window_size))
  local_indices = jnp.where(local_pos >= 0, local_indices, -1)
  indices = jnp.concatenate([compressed_indices, local_indices], axis=-1)
  lengths = jnp.full((batch, q_len), indexer_topk + window_size, dtype=jnp.int32)

  out = dsa_sparse_attention_batched(
      q.astype(jnp.bfloat16),
      kv.astype(jnp.bfloat16),
      indices,
      lengths,
      sinks.astype(jnp.float32),
      batch_axes=_CSA_BATCH_AXES,
      indexer_topk=indexer_topk,
      softmax_scale=1.0,
  )
  return out


__all__ = ["compress_ratio4", "indexer_ratio4", "sparse_attention_ratio4"]
