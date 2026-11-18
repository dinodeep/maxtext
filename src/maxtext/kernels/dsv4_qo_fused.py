# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Fused Triton kernels for the DeepSeek-V4 compressed-attention query / output paths.

The stock XLA lowering of the DSv4 query path (per-head unweighted RMSNorm -> partial interleaved RoPE on the
last 64 of 512 channels via slice + concatenate -> * softmax_scale) and of the output path (inverse RoPE via
concatenate -> grouped `...gi,gio->...go` projection) makes ~9 full passes over [B, S, 64, 512] bf16 tensors
per layer (fwd + remat + bwd). These kernels do the same math in fewer passes:

* `q_norm_rope_scale(q_up, cos, sin)`: one fused kernel forward (RMSNorm + RoPE + scale, bit-identical to the
  XLA formulation up to the fp32 sum order of the mean of squares) and one fused kernel backward (reads dq and
  q_up, writes d q_up), instead of 2 + 3 XLA passes.
* `inv_rope_grouped_proj(o, cos, sin, w)`: the forward keeps the stock XLA inverse RoPE + einsum, but the backward
  computes the input gradient of the grouped einsum in its natural [g, b, s, i] layout and applies the transposed
  inverse RoPE while writing [b, s, h, d] in one kernel, instead of XLA's transposing pad/add fusion.

Kernels are launched through Transformer Engine's Triton-JAX bridge and run per batch shard inside shard_map.
Launch config: 2 rows of 512 channels per program, 1 warp (fastest of the swept configs on GB300);
DSV4_QO_CFG="rows,warps[;rows,warps...]" overrides it (several configs = runtime autotuning).
Note: `triton` must be imported before `maxtext.common.checkpointing` (otherwise the triton import segfaults in
this container); MaxText's train entry point already does so.
"""

from __future__ import annotations

import functools
import os

import jax
import jax.numpy as jnp
import numpy as np
from jax.ad_checkpoint import checkpoint_name
from jax.interpreters import mlir
from jax.sharding import PartitionSpec as P

try:
  import triton
  import triton.language as tl
except ImportError:  # pragma: no cover - TPU / CPU installs; the flags then fail at trace time.
  triton = None

_BATCH_AXES = ("data", "fsdp", "expert")


# --------------------------------------------------------------------------------------------- kernels
if triton is not None:
  # Rows of D channels are loaded as [BLOCK_R, D // 2, 2] and split into even / odd channels, so the interleaved
  # rotary pairs (2i, 2i + 1) are available to each thread without a gather; loads and stores stay vectorized.
  @triton.jit
  def _q_fwd_kernel(
      x_ptr,
      cos_ptr,
      sin_ptr,
      out_ptr,
      N_ROWS: tl.constexpr,
      H: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      EPS: tl.constexpr,
      SCALE: tl.constexpr,
      IDX: tl.constexpr,
      BLOCK_R: tl.constexpr,
  ):
    """Forward: per-head RMSNorm, interleaved RoPE on the trailing channels, then the softmax scale."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    pr = tl.arange(0, D // 2)
    rmask2 = (rows < N_ROWS)[:, None]
    offs = rows.to(IDX)[:, None, None] * D + pr[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
    x = tl.load(x_ptr + offs, mask=rmask2[:, :, None], other=0.0).to(tl.float32)
    xe, xo = tl.split(x)
    r = tl.math.rsqrt(tl.sum(tl.reshape(x * x, (BLOCK_R, D)), axis=1) / D + EPS)[:, None]
    ye = (xe * r).to(tl.bfloat16).to(tl.float32)
    yo = (xo * r).to(tl.bfloat16).to(tl.float32)
    half = (D - NOPE) // 2
    is_rope = (pr >= NOPE // 2)[None, :]
    j = tl.maximum(pr - NOPE // 2, 0)[None, :]
    tok = (rows // H)[:, None]
    c = tl.load(cos_ptr + tok * half + j, mask=rmask2 & is_rope, other=1.0)
    s = tl.load(sin_ptr + tok * half + j, mask=rmask2 & is_rope, other=0.0)
    ze = tl.where(is_rope, (ye * c + (-yo) * s).to(tl.bfloat16).to(tl.float32), ye)
    zo = tl.where(is_rope, (yo * c + ye * s).to(tl.bfloat16).to(tl.float32), yo)
    out = tl.join((ze * SCALE).to(tl.bfloat16), (zo * SCALE).to(tl.bfloat16))
    tl.store(out_ptr + offs, out, mask=rmask2[:, :, None])

  @triton.jit
  def _q_bwd_kernel(
      g_ptr,
      x_ptr,
      cos_ptr,
      sin_ptr,
      dx_ptr,
      N_ROWS: tl.constexpr,
      H: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      EPS: tl.constexpr,
      SCALE: tl.constexpr,
      IDX: tl.constexpr,
      BLOCK_R: tl.constexpr,
  ):
    """Backward of _q_fwd_kernel: reads dq and q_up, writes d q_up."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    pr = tl.arange(0, D // 2)
    rmask2 = (rows < N_ROWS)[:, None]
    offs = rows.to(IDX)[:, None, None] * D + pr[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
    x = tl.load(x_ptr + offs, mask=rmask2[:, :, None], other=0.0).to(tl.float32)
    xe, xo = tl.split(x)
    r = tl.math.rsqrt(tl.sum(tl.reshape(x * x, (BLOCK_R, D)), axis=1) / D + EPS)[:, None]
    g = tl.load(g_ptr + offs, mask=rmask2[:, :, None], other=0.0).to(tl.float32)
    ge, go = tl.split(g)
    # d(z * scale) with a bf16 multiply
    gze = (ge * SCALE).to(tl.bfloat16).to(tl.float32)
    gzo = (go * SCALE).to(tl.bfloat16).to(tl.float32)
    half = (D - NOPE) // 2
    is_rope = (pr >= NOPE // 2)[None, :]
    j = tl.maximum(pr - NOPE // 2, 0)[None, :]
    tok = (rows // H)[:, None]
    c = tl.load(cos_ptr + tok * half + j, mask=rmask2 & is_rope, other=1.0)
    s = tl.load(sin_ptr + tok * half + j, mask=rmask2 & is_rope, other=0.0)
    # transpose of the rotation: (R^T w)_even = w_odd, (R^T w)_odd = -w_even
    gye = tl.where(is_rope, (gze * c + gzo * s).to(tl.bfloat16).to(tl.float32), gze)
    gyo = tl.where(is_rope, (gzo * c - gze * s).to(tl.bfloat16).to(tl.float32), gzo)
    # unweighted RMSNorm backward: dx = r * gy - x * r^3 * mean(gy * x)
    dot = ((tl.sum(gye * xe, axis=1) + tl.sum(gyo * xo, axis=1)) / D)[:, None]
    r3 = r * r * r
    dxe = r * gye - xe * r3 * dot
    dxo = r * gyo - xo * r3 * dot
    tl.store(dx_ptr + offs, tl.join(dxe.to(tl.bfloat16), dxo.to(tl.bfloat16)), mask=rmask2[:, :, None])

  @triton.jit
  def _o_bwd_kernel(
      dg_ptr,
      cos_ptr,
      sin_ptr,
      do_ptr,
      N_ROWS: tl.constexpr,
      BT: tl.constexpr,
      H: tl.constexpr,
      HG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      IDX: tl.constexpr,
      BLOCK_R: tl.constexpr,
  ):
    """dO[bt, h, :] = R^T(dOrot) with dOrot read from the grouped-GEMM gradient laid out [g, bt, (h % HG) * D + d]."""
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)  # row = bt * H + h
    pr = tl.arange(0, D // 2)
    rmask2 = (rows < N_ROWS)[:, None]
    bt = (rows // H).to(IDX)
    h = rows % H
    src = ((h // HG).to(IDX) * BT + bt) * (HG * D) + (h % HG).to(IDX) * D
    pair = pr[None, :, None] * 2 + tl.arange(0, 2)[None, None, :]
    g = tl.load(dg_ptr + src[:, None, None] + pair, mask=rmask2[:, :, None], other=0.0).to(tl.float32)
    ge, go = tl.split(g)
    half = (D - NOPE) // 2
    is_rope = (pr >= NOPE // 2)[None, :]
    j = tl.maximum(pr - NOPE // 2, 0)[None, :]
    c = tl.load(cos_ptr + bt[:, None] * half + j, mask=rmask2 & is_rope, other=1.0)
    s = -tl.load(sin_ptr + bt[:, None] * half + j, mask=rmask2 & is_rope, other=0.0)  # inverse rotation
    de = tl.where(is_rope, (ge * c + go * s).to(tl.bfloat16), ge.to(tl.bfloat16))
    do = tl.where(is_rope, (go * c - ge * s).to(tl.bfloat16), go.to(tl.bfloat16))
    dst = rows.to(IDX)[:, None, None] * D + pair
    tl.store(do_ptr + dst, tl.join(de, do), mask=rmask2[:, :, None])


# --------------------------------------------------------------------------------------------- primitives
def _make_primitive(name, kernel, abstract, grid_fn, constexpr_fn, num_warps=4):
  """Wraps a Triton kernel as a JAX primitive lowered through Transformer Engine's Triton bridge."""
  try:
    from jax.extend.core import Primitive  # pylint: disable=import-outside-toplevel
  except ImportError:  # pragma: no cover
    from jax.core import Primitive  # pylint: disable=import-outside-toplevel
  from transformer_engine.jax.triton_extensions import utils as te_triton  # pylint: disable=import-outside-toplevel
  from jax._src import dispatch  # pylint: disable=import-outside-toplevel

  kernel = triton.autotune(configs=_configs(), key=[])(kernel)
  prim = Primitive(name)
  prim.def_abstract_eval(abstract)
  prim.def_impl(functools.partial(dispatch.apply_primitive, prim))

  def lowering(ctx, *args, **params):
    return te_triton.triton_call_lowering(
        ctx, kernel, *args, grid=grid_fn(ctx, **params), constexprs=constexpr_fn(ctx, **params)
    )

  mlir.register_lowering(prim, lowering, platform="cuda")
  return prim


_PRIMS = {}


def _idx_type(numel):
  return tl.int32 if numel < 2**31 else tl.int64


def _configs():
  """Launch configs; DSV4_QO_CFG='rows,warps;rows,warps' overrides (first one wins without autotuning)."""
  spec = os.environ.get("DSV4_QO_CFG", "2,1")
  return [
      triton.Config({"BLOCK_R": int(r)}, num_warps=int(w), num_stages=1)
      for r, w in (c.split(",") for c in spec.split(";"))
  ]


def _prims():
  """Builds (once) the JAX primitives of the q forward, q backward and O backward kernels."""
  if _PRIMS:
    return _PRIMS
  if triton is None:
    raise ImportError("dsv4_fused_q / dsv4_fused_o_bwd need Triton (GPU) and Transformer Engine's Triton bridge")
  core = jax.core

  def q_abs(x, cos, sin, **_):
    return core.ShapedArray(x.shape, x.dtype)

  def q_grid(ctx, **_):
    n = int(np.prod(ctx.avals_in[0].shape[:-1]))
    return lambda meta: (triton.cdiv(n, meta["BLOCK_R"]),)

  def q_const(ctx, *, heads, nope, eps, scale):
    n = int(np.prod(ctx.avals_in[0].shape[:-1]))
    d = ctx.avals_in[0].shape[-1]
    return {"N_ROWS": n, "H": heads, "D": d, "NOPE": nope, "EPS": eps, "SCALE": scale, "IDX": _idx_type(n * d)}

  _PRIMS["q_fwd"] = _make_primitive("dsv4_q_norm_rope_fwd", _q_fwd_kernel, q_abs, q_grid, q_const)

  def qb_abs(g, x, cos, sin, **_):
    return core.ShapedArray(x.shape, x.dtype)

  _PRIMS["q_bwd"] = _make_primitive("dsv4_q_norm_rope_bwd", _q_bwd_kernel, qb_abs, q_grid, q_const)

  def ob_abs(dg, cos, sin, *, heads, nope):
    g, b, s, i = dg.shape
    return core.ShapedArray((b, s, heads, i * g // heads), dg.dtype)

  def ob_grid(ctx, *, heads, nope):
    del nope
    _, b, s, _ = ctx.avals_in[0].shape
    return lambda meta: (triton.cdiv(b * s * heads, meta["BLOCK_R"]),)

  def ob_const(ctx, *, heads, nope):
    g, b, s, i = ctx.avals_in[0].shape
    d = g * i // heads
    return {
        "N_ROWS": b * s * heads,
        "BT": b * s,
        "H": heads,
        "HG": heads // g,
        "D": d,
        "NOPE": nope,
        "IDX": _idx_type(b * s * heads * d),
    }

  _PRIMS["o_bwd"] = _make_primitive("dsv4_o_inv_rope_bwd", _o_bwd_kernel, ob_abs, ob_grid, ob_const)
  return _PRIMS


def _leading_batch_axis():
  from transformer_engine.jax.deepseek_v4 import _batch_axes_spec  # pylint: disable=import-outside-toplevel

  return _batch_axes_spec(_BATCH_AXES, 1)[0]


def _shard(fn, in_specs, out_spec):
  from transformer_engine.jax.deepseek_v4 import _batched_shard_map  # pylint: disable=import-outside-toplevel

  return _batched_shard_map(fn, in_specs, out_spec, _BATCH_AXES)


# --------------------------------------------------------------------------------------------- q path
def _bf16_scale(scale):
  return float(np.asarray(scale, dtype=jnp.bfloat16).astype(np.float32))


@functools.partial(jax.custom_vjp, nondiff_argnums=(3, 4))
def q_norm_rope_scale(q_up, cos, sin, eps, scale):
  """bf16(rope(bf16(rmsnorm(q_up))) * bf16(scale)) over [B, S, H, D] with cos/sin [B, S, rope_dim // 2] (fp32)."""
  return _q_fwd(q_up, cos, sin, eps, scale)


def _q_fwd(q_up, cos, sin, eps, scale):
  _, _, h, d = q_up.shape
  nope = d - 2 * cos.shape[-1]
  ax = _leading_batch_axis()

  def body(x, c, sn):
    return _prims()["q_fwd"].bind(x, c, sn, heads=h, nope=nope, eps=float(eps), scale=_bf16_scale(scale))

  return _shard(body, (P(ax, None, None, None), P(ax, None, None), P(ax, None, None)), P(ax, None, None, None))(
      q_up, cos.astype(jnp.float32), sin.astype(jnp.float32)
  )


def _q_vjp_fwd(q_up, cos, sin, eps, scale):
  return _q_fwd(q_up, cos, sin, eps, scale), (q_up, cos, sin)


def _q_vjp_bwd(eps, scale, res, g):
  """Fused backward of q_norm_rope_scale."""
  q_up, cos, sin = res
  _, _, h, d = q_up.shape
  nope = d - 2 * cos.shape[-1]
  ax = _leading_batch_axis()

  def body(gg, x, c, sn):
    return _prims()["q_bwd"].bind(gg, x, c, sn, heads=h, nope=nope, eps=float(eps), scale=_bf16_scale(scale))

  spec4, spec3 = P(ax, None, None, None), P(ax, None, None)
  dx = _shard(body, (spec4, spec4, spec3, spec3), spec4)(
      g.astype(q_up.dtype), q_up, cos.astype(jnp.float32), sin.astype(jnp.float32)
  )
  return dx, jnp.zeros_like(cos), jnp.zeros_like(sin)


q_norm_rope_scale.defvjp(_q_vjp_fwd, _q_vjp_bwd)


# --------------------------------------------------------------------------------------------- o path
def _inv_rope_xla(o, cos, sin):
  """Stock MaxText inverse interleaved RoPE on the trailing 2 * cos.shape[-1] channels (bit-identical).

  (Writing it as an in-place dynamic-update-slice of the 64 rotated channels was tried and is slower: XLA emits a
  full-size DUS loop fusion plus a copy, +2.5 ms per layer.)
  """
  from maxtext.layers.embeddings import _apply_rotary_pos_emb  # pylint: disable=import-outside-toplevel

  return _apply_rotary_pos_emb(o, cos, -sin, unsqueeze_dim=-2)


@functools.partial(jax.custom_vjp, nondiff_argnums=(4,))
def inv_rope_grouped_proj(o, cos, sin, w, precision):
  """einsum('...gi,gio->...go', inv_rope(o).reshape(b, s, g, i), w) for o [B, S, H, D], w [G, I, O]."""
  return _o_fwd(o, cos, sin, w, precision)[0]


def _o_fwd(o, cos, sin, w, precision):
  b, s, h, d = o.shape
  g = w.shape[0]
  o_rot = checkpoint_name(_inv_rope_xla(o, cos, sin), "attention_out")
  y = jnp.einsum("bsgi,gio->bsgo", o_rot.reshape(b, s, g, h * d // g), w.astype(o.dtype), precision=precision)
  return y, o_rot


def _o_vjp_fwd(o, cos, sin, w, precision):
  y, o_rot = _o_fwd(o, cos, sin, w, precision)
  return y, (o_rot, cos, sin, w)


def _o_vjp_bwd(precision, res, dy):
  """Backward of inv_rope_grouped_proj: natural-layout dgrad, then the fused transposed inverse RoPE."""
  o_rot, cos, sin, w = res
  b, s, h, d = o_rot.shape
  g = w.shape[0]
  dy = dy.astype(o_rot.dtype)
  wc = w.astype(o_rot.dtype)
  dw = jnp.einsum("bsgi,bsgo->gio", o_rot.reshape(b, s, g, h * d // g), dy, precision=precision).astype(w.dtype)
  # input gradient in the GEMM's natural batch-major layout [g, b, s, i]
  dg = jnp.einsum("bsgo,gio->gbsi", dy, wc, precision=precision)
  nope = d - 2 * cos.shape[-1]
  ax = _leading_batch_axis()

  def body(dgl, c, sn):
    return _prims()["o_bwd"].bind(dgl, c, sn, heads=h, nope=nope)

  do = _shard(body, (P(None, ax, None, None), P(ax, None, None), P(ax, None, None)), P(ax, None, None, None))(
      dg, cos.astype(jnp.float32), sin.astype(jnp.float32)
  )
  return do, jnp.zeros_like(cos), jnp.zeros_like(sin), dw


inv_rope_grouped_proj.defvjp(_o_vjp_fwd, _o_vjp_bwd)
