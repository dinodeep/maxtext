# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MaxText shape adapters for Transformer Engine DeepSeek-V4 CSA/HCA APIs.

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


def compress_ratio128(kv: Any, gate: Any, position_bias: Any) -> Any:
  """Runs the differentiable ratio-128 HCA (non-overlapping) compressor, before RMSNorm/RoPE.

  Returns ``[batch, sequence // 128, width]``; a trailing partial window is dropped.
  """
  from transformer_engine.jax.deepseek_v4 import hca_compressor_batched  # pylint: disable=import-outside-toplevel

  _, sequence, width = kv.shape
  if sequence < 128:
    raise ValueError("cuDNN DSv4 HCA requires sequence length >= 128")
  if gate.shape != kv.shape or position_bias.shape != (128, width):
    raise ValueError(f"expected gate={kv.shape} and position_bias={(128, width)}")
  return hca_compressor_batched(
      kv.astype(jnp.bfloat16),
      gate.astype(jnp.bfloat16),
      position_bias.astype(jnp.float32),
      batch_axes=_CSA_BATCH_AXES,
      ratio=128,
  )


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


def indexer_top_k_ratio4(scores: Any, *, top_k: int) -> Any:
  """Selects the ``top_k`` highest-scoring causally valid compressed blocks per query with cuDNN.

  ``scores`` is ``[batch, q_len, compressed_len]``; query ``t`` may choose among its first
  ``(t + 1) // 4`` blocks (positions are assumed to be ``arange``, as in non-packed training).
  Returns ``[batch, q_len, top_k]`` int32 block ids: unsorted, ``-1`` where fewer than ``top_k`` blocks
  are valid. Ties go to the smaller block id, like ``jax.lax.top_k``.
  """
  from transformer_engine.jax.deepseek_v4 import dsa_indexer_top_k_batched  # pylint: disable=import-outside-toplevel

  return dsa_indexer_top_k_batched(
      jax.lax.stop_gradient(scores).astype(jnp.float32), batch_axes=_CSA_BATCH_AXES, top_k=top_k, ratio=4
  )


def indexer_scores_ratio4(q: Any, compressed_k: Any, weights: Any, *, softmax_scale: float) -> dict[str, Any]:
  """Dense ratio-4 indexer scores and row LSE for the fused cuDNN loss backward.

  Returns the kernel inputs in BSHD layout together with ``scores``/``lse`` so that
  :func:`indexer_kl_loss_ratio4` can differentiate the loss without recomputing the forward.
  """
  from transformer_engine.jax.deepseek_v4 import dsa_indexer_scores_batched  # pylint: disable=import-outside-toplevel

  q_bshd = jnp.transpose(q, (0, 2, 1, 3)).astype(jnp.bfloat16)
  k = compressed_k[:, :, None, :].astype(jnp.bfloat16)
  weights = weights.astype(jnp.bfloat16)
  scores, lse = dsa_indexer_scores_batched(
      q_bshd, k, weights, batch_axes=_CSA_BATCH_AXES, ratio=4, sm_scale=softmax_scale
  )
  return {"q": q_bshd, "k": k, "weights": weights, "scores": scores, "lse": lse, "sm_scale": softmax_scale}


def indexer_kl_loss_ratio4(indexer_inputs: dict[str, Any], target: Any, *, eps: float) -> Any:
  """Sum over tokens of the dense indexer KL(target || softmax(scores)); backward runs in cuDNN.

  ``target`` is the constant ``[batch, q_len, compressed_len]`` teacher distribution. Gradients flow to
  the indexer's ``q``, ``k`` and ``weights`` held in ``indexer_inputs``.
  """
  from transformer_engine.jax.deepseek_v4 import dsa_indexer_kl_loss_batched  # pylint: disable=import-outside-toplevel

  shard_sums = dsa_indexer_kl_loss_batched(
      indexer_inputs["q"],
      indexer_inputs["k"],
      indexer_inputs["weights"],
      indexer_inputs["scores"],
      indexer_inputs["lse"],
      jax.lax.stop_gradient(target).astype(jnp.float32),
      batch_axes=_CSA_BATCH_AXES,
      ratio=4,
      sm_scale=indexer_inputs["sm_scale"],
      eps=eps,
  )
  return jnp.sum(shard_sums)


def _first_valid_blocks(valid: Any, num: int) -> tuple[Any, Any]:
  """Returns the ``num`` lowest-index valid blocks per row (ascending) and their validity, without a sort.

  Matches ``jax.lax.top_k`` over all-equal scores with invalid entries at ``-inf`` (ties -> lowest index).
  """
  csum = jnp.cumsum(valid.astype(jnp.int32), axis=-1)
  ranks = jnp.arange(1, num + 1, dtype=jnp.int32)
  selected = jax.vmap(jax.vmap(lambda c: jnp.searchsorted(c, ranks, side="left")))(csum).astype(jnp.int32)
  return selected, ranks <= csum[..., -1:]


def indexer_sparse_kl_loss_ratio4(indexer_inputs: dict[str, Any], target_selected: Any, *, eps: float) -> Any:
  """Sum over tokens of the sparse indexer KL(target || softmax over the selected blocks); backward runs in cuDNN.

  ``target_selected`` is the constant ``[batch, q_len, k]`` teacher gathered at ``indexer_inputs["topk_indices"]``
  (``-1`` = invalid slot, teacher ``0`` there). Gradients flow to the indexer's ``q``, ``k`` and ``weights``.
  """
  from transformer_engine.jax.deepseek_v4 import dsa_indexer_sparse_kl_loss_batched  # pylint: disable=import-outside-toplevel

  shard_sums = dsa_indexer_sparse_kl_loss_batched(
      indexer_inputs["q"],
      indexer_inputs["k"],
      indexer_inputs["weights"],
      indexer_inputs["scores"],
      indexer_inputs["topk_indices"].astype(jnp.int32),
      jax.lax.stop_gradient(target_selected).astype(jnp.float32),
      batch_axes=_CSA_BATCH_AXES,
      sm_scale=indexer_inputs["sm_scale"],
      eps=eps,
  )
  return jnp.sum(shard_sums)


def sparse_attention_ratio4(
    q: Any,
    local_kv: Any,
    compressed_kv: Any,
    compressed_mask: Any,
    sinks: Any,
    *,
    indexer_topk: int = 512,
    window_size: int = 128,
    topk_indices: Any = None,
) -> Any:
  """Runs combined 512 compressed + 128 local-token sparse attention.

  By default the selected compressed blocks are recovered from ``compressed_mask`` (the lowest-index
  ``indexer_topk`` blocks it leaves valid). If ``topk_indices`` (``[batch, q_len, k]`` int32, ``-1`` = invalid,
  the indexer's own selection) is given, they are used directly, restricted to blocks that ``compressed_mask``
  also leaves valid; this skips the mask round trip and its second top-k.
  """
  from transformer_engine.jax.deepseek_v4 import dsa_sparse_attention_batched  # pylint: disable=import-outside-toplevel

  batch, q_len, heads, head_dim = q.shape
  if (heads, head_dim, indexer_topk, window_size) != (64, 512, 512, 128):
    raise ValueError("cuDNN DSv4 sparse attention requires H=64, D=512, top-k=512, window=128")
  if local_kv.shape != (batch, q_len, 1, head_dim):
    raise ValueError("cuDNN DSv4 training requires self-attention with one local KV head")
  comp_len = compressed_kv.shape[1]
  mask = compressed_mask[:, 0] if compressed_mask.ndim == 4 else compressed_mask
  valid = mask > -1.0e20
  if topk_indices is None:
    selected, selected_valid = _first_valid_blocks(valid, indexer_topk)
  else:
    selected = topk_indices.astype(jnp.int32)
    selected_valid = (selected >= 0) & jnp.take_along_axis(valid, jnp.maximum(selected, 0), axis=-1)
    # Ascending order (invalid last) is what the mask path produced; the kernel accumulates in index order, so
    # this keeps the output bit-identical to it (top-k order alone differs by ~1 bf16 ulp, enough to show up
    # as a ~0.5% indexer-loss drift over 30 proxy steps).
    big = jnp.int32(1 << 30)
    selected = jnp.sort(jnp.where(selected_valid, selected, big), axis=-1)
    selected_valid = selected < big
    selected = jnp.where(selected_valid, selected, -1)
    pad = indexer_topk - selected.shape[-1]
    if pad > 0:
      selected = jnp.pad(selected, ((0, 0), (0, 0), (0, pad)), constant_values=-1)
      selected_valid = jnp.pad(selected_valid, ((0, 0), (0, 0), (0, pad)), constant_values=False)

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


def sparse_attention_hca(
    q: Any,
    local_kv: Any,
    compressed_kv: Any,
    sinks: Any,
    *,
    compress_ratio: int = 128,
    window_size: int = 128,
) -> Any:
  """Runs HCA attention (sliding window + every causally complete compressed block).

  Equivalent to ``HCAStaticMask``: query ``t`` sees local tokens ``t - window + 1 .. t``
  and compressed block ``c`` iff ``c < (t + 1) // compress_ratio``. The index list is
  fixed, so the kernel runs without an indexer prefix (``indexer_topk=0``).
  """
  from transformer_engine.jax.deepseek_v4 import dsa_sparse_attention_batched  # pylint: disable=import-outside-toplevel

  batch, q_len, heads, head_dim = q.shape
  if (heads, head_dim) != (64, 512):
    raise ValueError("cuDNN DSv4 sparse attention requires H=64, D=512")
  if window_size <= 0:
    raise ValueError("cuDNN DSv4 HCA requires a positive sliding window")
  if local_kv.shape != (batch, q_len, 1, head_dim):
    raise ValueError("cuDNN DSv4 training requires self-attention with one local KV head")
  comp_len = compressed_kv.shape[1]
  kv = jnp.concatenate([local_kv, compressed_kv], axis=1)[:, :, 0, :]

  query_pos = jnp.arange(q_len, dtype=jnp.int32)[:, None]
  local_pos = query_pos + jnp.arange(1 - window_size, 1, dtype=jnp.int32)[None, :]
  local_indices = jnp.where(local_pos >= 0, local_pos, -1)
  comp_pos = jnp.arange(comp_len, dtype=jnp.int32)[None, :]
  num_visible = jnp.minimum((query_pos + 1) // compress_ratio, comp_len)
  compressed_indices = jnp.where(comp_pos < num_visible, q_len + comp_pos, -1)
  indices = jnp.concatenate([local_indices, compressed_indices], axis=-1)
  # The kernel's physical K must be a multiple of 64.
  num_selected = window_size + comp_len
  padded = -(-num_selected // 64) * 64
  if padded > num_selected:
    indices = jnp.pad(indices, ((0, 0), (0, padded - num_selected)), constant_values=-1)
  indices = jnp.broadcast_to(indices[None], (batch, q_len, padded))
  # Visible compressed blocks form a prefix, so trailing entries can be skipped.
  lengths = jnp.broadcast_to((window_size + num_visible[:, 0])[None], (batch, q_len)).astype(jnp.int32)

  out = dsa_sparse_attention_batched(
      q.astype(jnp.bfloat16),
      kv.astype(jnp.bfloat16),
      indices,
      lengths,
      sinks.astype(jnp.float32),
      batch_axes=_CSA_BATCH_AXES,
      indexer_topk=0,
      softmax_scale=1.0,
  )
  return out


__all__ = [
    "compress_ratio4",
    "compress_ratio128",
    "indexer_ratio4",
    "indexer_scores_ratio4",
    "indexer_kl_loss_ratio4",
    "sparse_attention_ratio4",
    "sparse_attention_hca",
]
