# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Prepared THD caller integration; not packed CompressedAttention training."""

# Keep optional GPU dependencies behind the device checks.
# pylint: disable=import-outside-toplevel

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from maxtext.kernels import dsv4_cudnn


@pytest.mark.gpu_only
@pytest.mark.parametrize("ratio", [4, 128])
def test_prepared_thd_matches_independent_sequences(ratio):
  if not any(device.platform == "gpu" for device in jax.devices()):
    pytest.skip("requires an SM100 JAX device")
  from cudnn.tensor_adapter import get_compute_capability

  if get_compute_capability()[0] < 10:
    pytest.skip("requires SM100")
  from transformer_engine.jax.attention import SequenceDescriptor
  from transformer_engine.jax.deepseek_v4 import csa_compressor, dsa_sparse_attention
  from maxtext.layers.embeddings import DeepSeekV4RotaryEmbedding

  lengths = (131, 257)
  coff = 2 if ratio == 4 else 1
  rope = DeepSeekV4RotaryEmbedding(head_dim=512, fprop_dtype=jnp.bfloat16)
  index_rope = DeepSeekV4RotaryEmbedding(head_dim=128, partial_rotary_factor=0.5, fprop_dtype=jnp.bfloat16)
  key = jax.random.key(31)
  shapes = [
      (sum(lengths), 64, 512),
      (sum(lengths), 512),
      (sum(lengths), coff * 512),
      (sum(lengths), coff * 512),
      (ratio, coff * 512),
      (sum(lengths), 64, 128),
      (sum(lengths), 64),
      (64,),
  ]
  args = tuple(
      jax.random.normal(jax.random.fold_in(key, i), shape, jnp.float32 if i in (4, 7) else jnp.bfloat16) * 0.1
      for i, shape in enumerate(shapes)
  )

  def run(seqlens, query, local, values, gates, ape, index_q, weights, sinks):
    counts = tuple(n // ratio for n in seqlens)
    cu = jnp.array(np.cumsum((0,) + seqlens), jnp.int32)
    cu_comp = jnp.array(np.cumsum((0,) + counts), jnp.int32)
    capacity = sum(seqlens) // ratio
    compressed = csa_compressor(values, gates, ape, cu, cu_comp, total_comp=capacity, ratio=ratio, coff=coff)
    # The compressor does not promise initialized capacity padding.
    valid_comp = jnp.arange(capacity) < cu_comp[-1]
    compressed = jnp.where(valid_comp[:, None], compressed, 0)
    compressed = (
        compressed.astype(jnp.float32)
        * jax.lax.rsqrt(jnp.mean(jnp.square(compressed.astype(jnp.float32)), axis=-1, keepdims=True) + 1e-6)
    ).astype(jnp.bfloat16)
    positions = jnp.concatenate([jnp.arange(n) for n in seqlens])
    comp_positions = jnp.pad(jnp.concatenate([jnp.arange(n) * ratio for n in counts]), (0, capacity - sum(counts)))
    query = rope(query[None], positions[None], unsqueeze_dim=2)[0]
    local = rope(local[None], positions[None], unsqueeze_dim=None)[0]
    compressed = rope(compressed[None], comp_positions[None], unsqueeze_dim=None)[0]
    seg = jnp.repeat(jnp.arange(len(seqlens)), jnp.array(seqlens), total_repeat_length=sum(seqlens))
    auxiliary = jnp.array(0.0, jnp.float32)
    if ratio == 4:
      index_k = csa_compressor(
          values[:, :256], gates[:, :256], ape[:, :256], cu, cu_comp, total_comp=capacity, ratio=4, coff=2
      )
      index_k = jnp.where(valid_comp[:, None], index_k, 0)
      index_k = index_rope(index_k[None], comp_positions[None], unsqueeze_dim=None)[0]
      index_q = index_rope(index_q[None], positions[None], unsqueeze_dim=2)[0]
      descriptor = SequenceDescriptor.from_seqlens_and_offsets(
          (jnp.array([seqlens], jnp.int32), jnp.array([counts], jnp.int32)),
          (cu[None], cu_comp[None]),
      )
      # Exercise the actual MaxText adapter's THD branch, including its VJP.
      scores = dsv4_cudnn.indexer_ratio4(
          index_q,
          index_k,
          weights,
          softmax_scale=128**-0.5,
          sequence_descriptor=descriptor,
          max_seqlen_q=max(seqlens),
          max_seqlen_k=(max(seqlens) + 3) // 4,
      )
      chosen_scores, chosen = jax.lax.top_k(scores, 4)
      valid = jnp.isfinite(chosen_scores)
      auxiliary = jnp.where(jnp.isfinite(scores), scores, 0).sum() * 1e-4
    else:
      chosen = jnp.broadcast_to(jnp.arange(max(counts)), (sum(seqlens), max(counts)))
      valid = chosen < jnp.array(counts)[seg, None]
    valid &= (chosen + 1) * ratio <= positions[:, None] + 1
    # Score columns are segment-local; sparse IDs index concatenated packed KV.
    comp_ids = sum(seqlens) + cu_comp[seg, None] + chosen
    comp_ids = jnp.where(valid, comp_ids, -1)
    local_pos = positions[:, None] + jnp.arange(-15, 1)[None]
    local_ids = jnp.where(local_pos >= 0, cu[seg, None] + local_pos, -1)
    indices = jnp.concatenate((local_ids, comp_ids), axis=-1).astype(jnp.int32)
    width = (indices.shape[1] + 63) // 64 * 64
    indices = jnp.pad(indices, ((0, 0), (0, width - indices.shape[1])), constant_values=-1)
    out = dsa_sparse_attention(
        query,
        jnp.concatenate((local, compressed)),
        indices,
        jnp.full((sum(seqlens),), width, jnp.int32),
        sinks.astype(jnp.float32),
        indexer_topk=0,
        softmax_scale=512**-0.5,
    )
    return out, auxiliary

  def packed(*xs):
    return run(lengths, *xs)

  def separate(*xs):
    outputs, losses = [], []
    start = 0
    for n in lengths:
      sliced = tuple(x if i in (4, 7) else x[start : start + n] for i, x in enumerate(xs))
      output, aux = run((n,), *sliced)
      outputs.append(output)
      losses.append(aux)
      start += n
    return jnp.concatenate(outputs), sum(losses)

  def loss(fn, *xs):
    output, auxiliary = fn(*xs)
    return output.astype(jnp.float32).sum() + auxiliary

  packed_out, packed_aux = jax.jit(packed)(*args)
  expected, expected_aux = jax.jit(separate)(*args)
  np.testing.assert_allclose(packed_out.astype(jnp.float32), expected.astype(jnp.float32), atol=2e-2, rtol=2e-2)
  np.testing.assert_allclose(packed_aux, expected_aux, atol=2e-5, rtol=2e-5)
  packed_grads = jax.jit(jax.grad(lambda *xs: loss(packed, *xs), argnums=tuple(range(8))))(*args)
  expected_grads = jax.jit(jax.grad(lambda *xs: loss(separate, *xs), argnums=tuple(range(8))))(*args)
  for actual, ref in zip(packed_grads, expected_grads):
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual.astype(jnp.float32), ref.astype(jnp.float32), atol=4e-2, rtol=2e-2)


@pytest.mark.parametrize("ratio", [4, 128])
def test_training_block_metadata_handles_unaligned_documents(ratio):
  from maxtext.layers.attention_compressed import compressed_block_metadata

  lengths = (3, 129, 256)
  # Padding positions stay zero and must not form an extra compressed block.
  positions = jnp.array(np.concatenate([np.arange(n) for n in lengths] + [np.zeros(9)]), jnp.int32)[None]
  starts, local_positions, valid = jax.jit(lambda p: compressed_block_metadata(p, ratio))(positions)
  expected_starts, expected_positions = [], []
  offset = 0
  for length in lengths:
    expected_starts.extend(offset + np.arange(length // ratio) * ratio)
    expected_positions.extend(np.arange(length // ratio) * ratio)
    offset += length
  count = len(expected_starts)
  np.testing.assert_array_equal(starts[0, :count], expected_starts)
  np.testing.assert_array_equal(local_positions[0, :count], expected_positions)
  np.testing.assert_array_equal(valid[0], np.arange(positions.shape[1] // ratio) < count)
