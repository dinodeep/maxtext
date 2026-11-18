# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Faster SM100 DSA forward for DSv4 (H=64, D=512), as a drop-in for the cuDNN frontend binding.

`install_if_enabled()` (called when `dsv4_cudnn` is imported) replaces
`cudnn.sparse_attention_forward_jax_sm100` with `sparse_attention_forward_jax_sm100_fast` when DSV4_FAST_FWD=1.
Transformer Engine's DeepSeek-V4 wrappers import that function from `cudnn` at call time, so replacing the module
attribute is enough. The kernel (`dsv4_dsa_fwd_sm100.SparseAttentionForwardSm100Head64Persistent`) is a persistent
version of the cuDNN frontend 1.32 kernel. Calls with indexer_topk != 0 (indexer-prefix LSE requested, e.g. the
non-compact CSA layout) fall back to the stock kernel; with indexer_topk == 0 the fourth output (lse_indexer) is
not written (TE discards it).
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp

from maxtext.utils import max_logging

_kernel = None
_stock = None
_adapter = None


def _load_kernel():
  """Builds (once) the persistent forward kernel object."""
  global _kernel
  if _kernel is None:
    from maxtext.kernels import dsv4_dsa_fwd_sm100  # pylint: disable=import-outside-toplevel

    _kernel = dsv4_dsa_fwd_sm100.SparseAttentionForwardSm100Head64Persistent(head_dim=512, indexer_topk=0)
  return _kernel


def _adapter_factory():
  """CuTe JIT adapter from the cuDNN frontend JAX call convention to the kernel's argument order."""
  import cutlass  # pylint: disable=import-outside-toplevel
  from cutlass import cute  # pylint: disable=import-outside-toplevel

  @cute.jit
  def adapter(
      stream, q, kv, indices, topk_length, attn_sink, out, max_logits, lse, lse_indexer, *, kernel, softmax_scale
  ):
    kernel(
        q, kv, indices, out, max_logits, lse, lse_indexer, attn_sink, topk_length, cutlass.Float32(softmax_scale), stream
    )

  return adapter


def sparse_attention_forward_jax_sm100_fast(
    q, kv, topk_indices, topk_length, attn_sink, *, softmax_scale=None, indexer_topk=512
):
  """Same signature and outputs as cudnn.sparse_attention_forward_jax_sm100 (H=64, D=512, bf16)."""
  global _adapter
  if indexer_topk != 0:
    return _stock(q, kv, topk_indices, topk_length, attn_sink, softmax_scale=softmax_scale, indexer_topk=indexer_topk)
  if q.ndim != 3 or q.shape[1:] != (64, 512) or q.dtype != jnp.bfloat16:
    raise ValueError("q must have shape (Tq, 64, 512) and dtype bfloat16")
  if kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype:
    raise ValueError("kv must have shape (Tkv, 512) with q's dtype")
  if (
      topk_indices.ndim != 2
      or topk_indices.shape[0] != q.shape[0]
      or topk_indices.dtype != jnp.int32
      or topk_indices.shape[1] % 64
  ):
    raise ValueError("topk_indices must be (Tq, K) int32 with K % 64 == 0")
  if topk_length.shape != (q.shape[0],) or topk_length.dtype != jnp.int32:
    raise ValueError("topk_length must be (Tq,) int32")
  if attn_sink.shape != (64,) or attn_sink.dtype != jnp.float32:
    raise ValueError("attn_sink must be (64,) float32")
  from cudnn.jax import call  # pylint: disable=import-outside-toplevel

  if _adapter is None:
    _adapter = _adapter_factory()
  scale = 1.0 / math.sqrt(512) if softmax_scale is None else float(softmax_scale)
  tq = q.shape[0]
  return call(
      _adapter,
      output_shape_dtype=(
          jax.ShapeDtypeStruct((tq, 64, 512), q.dtype),
          jax.ShapeDtypeStruct((tq, 64), jnp.float32),
          jax.ShapeDtypeStruct((tq, 64), jnp.float32),
          jax.ShapeDtypeStruct((tq, 64), jnp.float32),
      ),
      kernel=_load_kernel(),
      softmax_scale=scale,
  )(q, kv, topk_indices, topk_length, attn_sink)


def install():
  """Replaces cudnn.sparse_attention_forward_jax_sm100 with the fast kernel (idempotent)."""
  global _stock
  import cudnn  # pylint: disable=import-outside-toplevel

  cur = cudnn.sparse_attention_forward_jax_sm100
  if cur is sparse_attention_forward_jax_sm100_fast:
    return
  _stock = cur
  cudnn.sparse_attention_forward_jax_sm100 = sparse_attention_forward_jax_sm100_fast
  try:
    import cudnn.deepseek_sparse_attention.jax_api as ja  # pylint: disable=import-outside-toplevel

    ja.sparse_attention_forward_jax_sm100 = sparse_attention_forward_jax_sm100_fast
  except ImportError:  # pragma: no cover
    pass
  max_logging.log("DSV4_FAST_FWD=1: cudnn.sparse_attention_forward_jax_sm100 -> persistent SM100 DSA forward kernel")


def install_if_enabled():
  """Calls install() when DSV4_FAST_FWD=1 (default off)."""
  if os.environ.get("DSV4_FAST_FWD", "0") == "1":
    install()
