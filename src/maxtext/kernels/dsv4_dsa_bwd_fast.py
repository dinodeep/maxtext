# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Faster SM100 DSA backward for DSv4 (H=64, D=512), as a drop-in for the cuDNN frontend binding.

`install_if_enabled()` (called when `dsv4_cudnn` is imported) replaces
`cudnn.sparse_attention_backward_jax_sm100` with `sparse_attention_backward_jax_sm100_fast` when DSV4_FAST_BWD=1.
Transformer Engine's DeepSeek-V4 wrappers import that function from `cudnn` at call time, so replacing the module
attribute is enough. Same inputs, outputs and launch sequence as the stock binding (clear dKV workspace, sum_OdO,
bwd, convert, sum_dSink); only the bwd kernel (`dsv4_dsa_bwd_sm100.FlashAttentionDSABackwardSm100`) is replaced.

DSV4_FAST_BWD_DQ_INIT selects how the dQ output buffer is initialized:
  zero (default, like the stock binding) | none (no fill: the kernel writes every dQ element, checked with a NaN
  fill) | nan (fill with NaN, to check that every element is written).
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp

from maxtext.utils import max_logging

_kernels = {}
_stock = None
_adapter = None


def _adapter_factory():
  """CuTe JIT adapter from the cuDNN frontend JAX call convention to the kernel's argument order."""
  import cutlass  # pylint: disable=import-outside-toplevel
  from cutlass import cute  # pylint: disable=import-outside-toplevel

  @cute.jit
  def adapter(
      stream,
      q,
      kv,
      out,
      dout,
      lse,
      attn_sink,
      indices,
      topk_length,
      dq,
      dkv,
      dsink,
      ws_lse,
      ws_dkv,
      *,
      kernel,
      softmax_scale,
      total_q,
      total_kv,
  ):
    ps = (cutlass.Int32(total_q), cutlass.Int32(total_kv), cutlass.Int32(512), (cutlass.Int32(64), cutlass.Int32(1)))
    kernel(
        ps,
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        indices,
        topk_length,
        dq,
        dkv,
        dsink,
        ws_lse,
        ws_dkv,
        cutlass.Float32(softmax_scale),
        stream,
    )

  return adapter


def _init_map(zeros_init):
  """Output initialization: dQ (output 0) per DSV4_FAST_BWD_DQ_INIT; dKV, dsink and the workspaces are zeroed."""
  mode = os.environ.get("DSV4_FAST_BWD_DQ_INIT", "zero")
  init = {1: zeros_init, 2: zeros_init, 3: zeros_init, 4: zeros_init}
  if mode == "zero":
    init[0] = zeros_init
  elif mode == "nan":
    init[0] = lambda sd: jnp.full(sd.shape, jnp.nan, sd.dtype)
  elif mode != "none":
    raise ValueError(f"DSV4_FAST_BWD_DQ_INIT must be zero, none or nan, got {mode!r}")
  return init


def sparse_attention_backward_jax_sm100_fast(
    q, kv, out, dout, lse, attn_sink, topk_indices, topk_length, *, softmax_scale=None
):
  """Same signature and outputs as cudnn.sparse_attention_backward_jax_sm100 (H=64, D=512, bf16)."""
  global _adapter
  if q.shape[1:] != (64, 512) or q.dtype != jnp.bfloat16:
    raise ValueError("q must be (Tq, 64, 512) bfloat16")
  if kv.ndim != 2 or kv.shape[1] != 512 or kv.dtype != q.dtype:
    raise ValueError("kv must be (Tkv, 512) bfloat16")
  if out.shape != q.shape or dout.shape != q.shape or out.dtype != q.dtype or dout.dtype != q.dtype:
    raise ValueError("out and dout must match q")
  if lse.shape != q.shape[:2] or lse.dtype != jnp.float32:
    raise ValueError("lse must be (Tq, 64) float32")
  if attn_sink.shape != (64,) or attn_sink.dtype != jnp.float32:
    raise ValueError("attn_sink must be (64,) float32")
  if topk_indices.shape[0] != q.shape[0] or topk_indices.dtype != jnp.int32:
    raise ValueError("topk_indices must be (Tq, K) int32")
  if topk_length.shape != (q.shape[0],) or topk_length.dtype != jnp.int32:
    raise ValueError("topk_length must be (Tq,) int32")
  import cutlass  # pylint: disable=import-outside-toplevel
  from cudnn.jax import call, zeros_init  # pylint: disable=import-outside-toplevel
  from maxtext.kernels.dsv4_dsa_bwd_sm100 import FlashAttentionDSABackwardSm100  # pylint: disable=import-outside-toplevel

  total_q, total_kv, max_topk = q.shape[0], kv.shape[0], topk_indices.shape[1]
  kern = _kernels.get(max_topk)
  if kern is None:
    kern = FlashAttentionDSABackwardSm100(
        element_dtype=cutlass.BFloat16, head_dim=512, head_dim_v=512, block_tile=64, max_topk=max_topk
    )
    _kernels[max_topk] = kern
  if _adapter is None:
    _adapter = _adapter_factory()
  # pylint: disable=protected-access
  ws_lse = FlashAttentionDSABackwardSm100._get_workspace_size_LSE_OdO(total_q, 512, 64, 1, cutlass.Float32)
  ws_dkv = FlashAttentionDSABackwardSm100._get_workspace_size_dKV(total_kv, 512, 1, cutlass.Float32)
  # pylint: enable=protected-access
  scale = 1.0 / math.sqrt(512) if softmax_scale is None else float(softmax_scale)
  res = call(
      _adapter,
      output_shape_dtype=(
          jax.ShapeDtypeStruct(q.shape, q.dtype),
          jax.ShapeDtypeStruct(kv.shape, kv.dtype),
          jax.ShapeDtypeStruct((64,), jnp.float32),
          jax.ShapeDtypeStruct(ws_lse, jnp.uint8),
          jax.ShapeDtypeStruct(ws_dkv, jnp.uint8),
      ),
      initialized_outputs=_init_map(zeros_init),
      kernel=kern,
      softmax_scale=scale,
      total_q=total_q,
      total_kv=total_kv,
  )(q, kv, out, dout, lse, attn_sink, topk_indices, topk_length)
  return res[:3]


def install():
  """Replaces cudnn.sparse_attention_backward_jax_sm100 with the fast kernel (idempotent)."""
  global _stock
  import cudnn  # pylint: disable=import-outside-toplevel

  cur = cudnn.sparse_attention_backward_jax_sm100
  if cur is sparse_attention_backward_jax_sm100_fast:
    return
  _stock = cur
  cudnn.sparse_attention_backward_jax_sm100 = sparse_attention_backward_jax_sm100_fast
  try:
    import cudnn.deepseek_sparse_attention.jax_api as ja  # pylint: disable=import-outside-toplevel

    ja.sparse_attention_backward_jax_sm100 = sparse_attention_backward_jax_sm100_fast
  except ImportError:  # pragma: no cover
    pass
  max_logging.log(
      "DSV4_FAST_BWD=1: cudnn.sparse_attention_backward_jax_sm100 -> persistent SM100 DSA backward kernel, dQ init: "
      + os.environ.get("DSV4_FAST_BWD_DQ_INIT", "zero")
  )


def install_if_enabled():
  """Calls install() when DSV4_FAST_BWD=1 (default off)."""
  if os.environ.get("DSV4_FAST_BWD", "0") == "1":
    install()
