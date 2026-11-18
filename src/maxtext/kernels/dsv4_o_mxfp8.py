# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""MXFP8 (TransformerEngine) versions of the DeepSeek-V4 grouped output projection `o_a_proj` (dsv4_o_a_proj_mxfp8).

`DeepSeekV4GroupedLinear` computes y[..., g, :] = x[..., g, :] @ w[g] for G (= o_groups = 8) equal groups, which is a
batched GEMM. TE/JAX has no batched MXFP8 GEMM (`dense` takes no batch dims), so the options are:

* impl "fused" (default; used in CompressedAttention's `dsv4_fused_o_bwd` branch), `inv_rope_grouped_proj_mxfp8_fused`:
  a Triton kernel reads the attention output in its natural [B, S, H, D] layout, applies the inverse RoPE and writes
  TE's group-major MXFP8 operands directly (rowwise + colwise e4m3 data, GEMM-swizzled e8m0 scales; bit-identical to
  TE's grouped_quantize of the transposed tensor). TE grouped GEMMs run fwd / dgrad / wgrad. In the backward the same
  kernel quantizes dy (no transpose needed), and the group-major dgrad feeds `dsv4_qo_fused`'s o-backward kernel.
* impl "grouped", `grouped_linear_mxfp8` / `inv_rope_grouped_proj_mxfp8`: XLA transposes to group-major + TE
  `grouped_dense` (TE quantizes). Simple, but slower than BF16 (transposes + separate quantize pass).
* impl "loop", `grouped_linear_mxfp8_loop`: G calls of TE MXFP8 `dense` on x[..., g, :] and w[g]. Simplest; slowest
  (strided slices + concat + 8 small quantizes).

MXFP8 quantizers carry no state (scales come from the tensor), so the quantizer sets are built directly with
`QuantizerFactory.create_set` (same recipe as `quantization=te_mxfp8` / `te_gmm_quantization=te_mxfp8`).
Measurements (GB300, per device B=7, S=4096): see job 3757408's REPORT.md.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_name
from jax.sharding import PartitionSpec as P


def _mx_set(n_groups=None):
  from transformer_engine.jax.quantize import QuantizerFactory, ScalingMode  # pylint: disable=import-outside-toplevel

  return QuantizerFactory.create_set(
      scaling_mode=ScalingMode.MXFP8_1D_SCALING,
      fwd_dtype=jnp.float8_e4m3fn,
      bwd_dtype=jnp.float8_e4m3fn,
      is_2x2x=True,
      n_groups=n_groups,
  )


def _qo():
  from maxtext.kernels import dsv4_qo_fused  # pylint: disable=import-outside-toplevel

  return dsv4_qo_fused


def grouped_linear_mxfp8_loop(x, w):
  """x [..., G, K], w [G, K, N] -> [..., G, N]; one TE MXFP8 dense per group."""
  from transformer_engine.jax.dense import dense  # pylint: disable=import-outside-toplevel

  g_axis = x.ndim - 2
  outs = [
      dense(jnp.take(x, g, axis=g_axis), w[g], contracting_dims=((x.ndim - 2,), (0,)), quantizer_set=_mx_set())
      for g in range(w.shape[0])
  ]
  return jnp.stack(outs, axis=g_axis)


def _grouped_gemm_gm(x_gm, w):
  """x_gm [G, b, s, K] (group-major), w [G, K, N] -> [G, b, s, N] via one TE MXFP8 grouped GEMM."""
  from transformer_engine.jax.dense import grouped_dense  # pylint: disable=import-outside-toplevel

  g, b, s, k = x_gm.shape
  group_sizes = jnp.full((g,), b * s, jnp.int32)
  y = grouped_dense(x_gm.reshape(g * b * s, k), w, group_sizes, contracting_dims=((1,), (1,)), quantizer_set=_mx_set(g))
  return y.reshape(g, b, s, w.shape[-1])


def grouped_linear_mxfp8(x, w):
  """x [B, S, G, K] (B sharded over the batch axes), w [G, K, N] -> [B, S, G, N]."""
  qo = _qo()
  ax = qo._leading_batch_axis()  # pylint: disable=protected-access

  def body(xl, wl):
    y = _grouped_gemm_gm(jnp.transpose(xl, (2, 0, 1, 3)), wl.astype(xl.dtype))
    return jnp.transpose(y, (1, 2, 0, 3))

  return qo._shard(body, (P(ax, None, None, None), P(None, None, None)), P(ax, None, None, None))(x, w)  # pylint: disable=protected-access


# ------------------------------------------------------------------ inverse RoPE -> group-major, fused backward
def inv_rope_grouped_proj_mxfp8(o, cos, sin, w):
  """MXFP8 version of dsv4_qo_fused.inv_rope_grouped_proj: o [B, S, H, D], w [G, I, O] -> [B, S, G, O]."""
  qo = _qo()
  ax = qo._leading_batch_axis()  # pylint: disable=protected-access
  _, s, h, d = o.shape
  g = w.shape[0]
  rope_half = cos.shape[-1]
  nope = d - 2 * rope_half

  @jax.custom_vjp
  def inv_rope_gm(ol, c, sn):
    o_rot = qo._inv_rope_xla(ol, c, sn)  # pylint: disable=protected-access
    bl = ol.shape[0]
    return jnp.transpose(o_rot.reshape(bl, s, g, h * d // g), (2, 0, 1, 3))

  def vjp_fwd(ol, c, sn):
    return inv_rope_gm(ol, c, sn), (c, sn)

  def vjp_bwd(res, dg):
    c, sn = res
    do = qo._prims()["o_bwd"].bind(dg.astype(o.dtype), c, sn, heads=h, nope=nope)  # pylint: disable=protected-access
    return do, jnp.zeros_like(c), jnp.zeros_like(sn)

  inv_rope_gm.defvjp(vjp_fwd, vjp_bwd)

  def body(ol, c, sn, wl):
    x_gm = checkpoint_name(inv_rope_gm(ol, c, sn), "attention_out")
    y = _grouped_gemm_gm(x_gm, wl.astype(ol.dtype))
    return jnp.transpose(y, (1, 2, 0, 3))

  spec4, spec3 = P(ax, None, None, None), P(ax, None, None)
  return qo._shard(body, (spec4, spec3, spec3, P(None, None, None)), spec4)(  # pylint: disable=protected-access
      o, cos.astype(jnp.float32), sin.astype(jnp.float32), w
  )


# ================================================================================================================
# Fused version: Triton kernels that read the natural (token-major) layout and write TE's group-major MXFP8
# operands (rowwise + colwise data, GEMM-swizzled e8m0 scales), so the grouped GEMMs need no XLA transpose and no
# separate TE quantize pass. For the o_a input the kernel also applies the inverse RoPE (it replaces the XLA
# inverse-RoPE fusion); the same kernel without RoPE quantizes the incoming gradient dy.
# ================================================================================================================
try:
  import triton
  import triton.language as tl
except ImportError:  # pragma: no cover
  triton = None

_TILE = 128  # tokens x channels per program; one program writes exactly one 512-byte swizzled scale tile per layout

if triton is not None:

  @triton.jit
  def _e8m0(amax):
    """TE's MXFP8 scale: biased e8m0 exponent of amax / 448 rounded up, and the exact power-of-2 multiplier."""
    v = amax * 0.002232142857142857  # amax * (1 / 448), as TE's CUDA kernel (max_norm_rcp)
    bits = v.to(tl.uint32, bitcast=True)
    exp = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    ru = (mant > 0) & (exp != 0xFE) & ~((exp == 0) & (mant <= 0x400000))
    exp = tl.where(ru, exp + 1, exp)
    mul = ((254 - exp) << 23).to(tl.float32, bitcast=True)  # 2^(127 - exp)
    mul = tl.where(exp == 254, 2.0**-127 * 1.0, mul)
    return exp.to(tl.uint8), mul

  @triton.jit
  def _to_e4m3(x):
    """fp32 -> e4m3 (round-to-nearest-even, saturating) as uint8; TE's lowering does not enable fp8e4nv in Triton."""
    return tl.inline_asm_elementwise(
        asm="{ .reg .b16 lo, hi; cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; "
        "mov.b32 $0, {lo, hi}; }",
        constraints="=r,f,f,f,f",
        args=[x],
        dtype=tl.uint8,
        is_pure=True,
        pack=4,
    )

  @triton.jit
  def _quant_gm_body(
      x_ptr,
      cos_ptr,
      sin_ptr,
      row_ptr,
      col_ptr,
      rs_ptr,
      cs_ptr,
      BT: tl.constexpr,
      G: tl.constexpr,
      KG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      HAS_ROPE: tl.constexpr,
      IDX: tl.constexpr,
      SLABS: tl.constexpr,
      WANT_ROW: tl.constexpr,
      WANT_COL: tl.constexpr,
      MODE: tl.constexpr = 0,
  ):
    """x [BT, G * KG] bf16 (token-major) -> group-major MXFP8 [G, BT, KG] rowwise and/or colwise data and scales.

    One program = SLABS x 32 tokens x 128 channels, processed as slabs of [32 tokens, 4 blocks, 32 channels] so both
    block reductions are plain axis reductions. Scales are written in TE's GEMM-swizzled layout:
    (R//128, 4, 32, C//4, 4) -> (0, 3, 2, 1, 4), per group, rows = tokens (rowwise) or channels (colwise).
    With HAS_ROPE the inverse interleaved RoPE is applied first (cos / sin given as [BT, 64] tables per 128-channel
    tile pair slot, see quant_gm), rounded to bf16 like the XLA formulation; only rotary tiles take that branch.
    """
    TILE: tl.constexpr = 128
    pt = tl.program_id(0)
    pc = tl.program_id(1)
    TPH: tl.constexpr = D // TILE  # channel tiles per head; the rotary channels live in the last one
    if MODE == 1:  # non-rotary tiles only
      c0 = ((pc // (TPH - 1)) * TPH + pc % (TPH - 1)) * TILE
    elif MODE == 2:  # rotary tiles only
      c0 = (pc * TPH + TPH - 1) * TILE
    else:
      c0 = pc * TILE  # global channel in [0, G * KG)
    g = c0 // KG
    k0 = c0 - g * KG  # channel within the group
    rr = tl.arange(0, 32)
    qq = tl.arange(0, 4)
    ww = tl.arange(0, 32)
    pp = tl.arange(0, 16)
    ch = qq[None, :, None] * 32 + ww[None, None, :]  # [1, 4, 32] channel within the tile
    if MODE == 1:
      rope_tile = False
    elif MODE == 2:
      rope_tile = True
    else:
      rope_tile = (c0 % D) + TILE > NOPE
    qp = qq[None, :, None] * 16 + pp[None, None, :]  # [1, 4, 16] rotary-pair slot within the tile
    for i in tl.static_range(SLABS):
      ts = (pt * SLABS + i) * 32
      a = ts // TILE
      bb = (ts % TILE) // 32
      rows = ts + rr
      src = rows.to(IDX)[:, None, None] * (G * KG) + c0 + ch
      if HAS_ROPE:
        if rope_tile:
          cs_off = rows.to(IDX)[:, None, None] * 64 + qp
          c = tl.load(cos_ptr + cs_off)
          s = tl.load(sin_ptr + cs_off)
          x = tl.load(x_ptr + src).to(tl.float32)
          # view as [32, 4, 16, 2] so each rotary pair is in-thread (tl.split), then back to 3-D
          xe, xo = tl.split(tl.reshape(x, (32, 4, 16, 2)))
          ze = (xe * c + xo * s).to(tl.bfloat16).to(tl.float32)
          zo = (xo * c - xe * s).to(tl.bfloat16).to(tl.float32)
          x = tl.reshape(tl.join(ze, zo), (32, 4, 32))
        else:
          x = tl.load(x_ptr + src).to(tl.float32)
      else:
        x = tl.load(x_ptr + src).to(tl.float32)
      ax = tl.abs(x)
      dst = (g.to(IDX) * BT + rows.to(IDX))[:, None, None] * KG + k0 + ch
      if WANT_ROW:
        r_exp, r_mul = _e8m0(tl.max(ax, axis=2))  # [32, 4]: blocks of 32 channels
        tl.store(row_ptr + dst, _to_e4m3(x * r_mul[:, :, None]))
        rbase = g.to(IDX) * BT * (KG // 32) + (a * (KG // 128) + k0 // TILE) * 512
        tl.store(rs_ptr + rbase + rr[:, None] * 16 + bb * 4 + qq[None, :], r_exp)
      if WANT_COL:
        c_exp, c_mul = _e8m0(tl.max(ax, axis=0))  # [4, 32]: blocks of 32 tokens
        tl.store(col_ptr + dst, _to_e4m3(x * c_mul[None, :, :]))
        cbase = g.to(IDX) * KG * (BT // 32) + ((k0 // TILE) * (BT // 128) + a) * 512
        tl.store(cs_ptr + cbase + ww[None, :] * 16 + qq[:, None] * 4 + bb, c_exp)


if triton is not None:

  @triton.jit
  def _quant_gm_kernel(
      x_ptr,
      cos_ptr,
      sin_ptr,
      row_ptr,
      col_ptr,
      rs_ptr,
      cs_ptr,
      BT: tl.constexpr,
      G: tl.constexpr,
      KG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      HAS_ROPE: tl.constexpr,
      IDX: tl.constexpr,
      SLABS: tl.constexpr,
  ):
    """All channel tiles, rowwise and colwise outputs in one launch."""
    _quant_gm_body(
        x_ptr, cos_ptr, sin_ptr, row_ptr, col_ptr, rs_ptr, cs_ptr, BT, G, KG, D, NOPE, HAS_ROPE, IDX, SLABS, True, True
    )

  @triton.jit
  def _quant_gm_norope_tiles_kernel(
      x_ptr,
      cos_ptr,
      sin_ptr,
      row_ptr,
      col_ptr,
      rs_ptr,
      cs_ptr,
      BT: tl.constexpr,
      G: tl.constexpr,
      KG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      HAS_ROPE: tl.constexpr,
      IDX: tl.constexpr,
      SLABS: tl.constexpr,
  ):
    """Non-rotary channel tiles only (first launch of the two-launch variant)."""
    _quant_gm_body(
        x_ptr, cos_ptr, sin_ptr, row_ptr, col_ptr, rs_ptr, cs_ptr, BT, G, KG, D, NOPE, HAS_ROPE, IDX, SLABS, True, True, 1
    )

  @triton.jit
  def _quant_gm_rope_tiles_kernel(
      x_ptr,
      cos_ptr,
      sin_ptr,
      row_in,
      col_in,
      rs_in,
      cs_in,
      row_ptr,
      col_ptr,
      rs_ptr,
      cs_ptr,
      BT: tl.constexpr,
      G: tl.constexpr,
      KG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      HAS_ROPE: tl.constexpr,
      IDX: tl.constexpr,
      SLABS: tl.constexpr,
  ):
    """Rotary channel tiles only (second launch; writes into the aliased outputs of the first)."""
    # outputs alias row_in / col_in / rs_in / cs_in (written by the non-rotary launch)
    _quant_gm_body(
        x_ptr, cos_ptr, sin_ptr, row_ptr, col_ptr, rs_ptr, cs_ptr, BT, G, KG, D, NOPE, HAS_ROPE, IDX, SLABS, True, True, 2
    )

  @triton.jit
  def _quant_gm_row_kernel(
      x_ptr,
      cos_ptr,
      sin_ptr,
      row_ptr,
      rs_ptr,
      BT: tl.constexpr,
      G: tl.constexpr,
      KG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      HAS_ROPE: tl.constexpr,
      IDX: tl.constexpr,
      SLABS: tl.constexpr,
  ):
    """Rowwise MXFP8 output only."""
    _quant_gm_body(
        x_ptr, cos_ptr, sin_ptr, row_ptr, row_ptr, rs_ptr, rs_ptr, BT, G, KG, D, NOPE, HAS_ROPE, IDX, SLABS, True, False
    )

  @triton.jit
  def _quant_gm_col_kernel(
      x_ptr,
      cos_ptr,
      sin_ptr,
      col_ptr,
      cs_ptr,
      BT: tl.constexpr,
      G: tl.constexpr,
      KG: tl.constexpr,
      D: tl.constexpr,
      NOPE: tl.constexpr,
      HAS_ROPE: tl.constexpr,
      IDX: tl.constexpr,
      SLABS: tl.constexpr,
  ):
    """Colwise MXFP8 output only."""
    _quant_gm_body(
        x_ptr, cos_ptr, sin_ptr, col_ptr, col_ptr, cs_ptr, cs_ptr, BT, G, KG, D, NOPE, HAS_ROPE, IDX, SLABS, False, True
    )


def _grouped_scale_size(m_total, k, g, colwise):
  """Size of TE's padded grouped MXFP8 scale buffer (ScalingMode.get_grouped_scale_shape)."""

  def divup(a, b):
    return -(-a // b)

  if colwise:
    nx, ny, ax, ay = m_total // 32, k, 4, 128
  else:
    nx, ny, ax, ay = m_total, k // 32, 128, 4
  return ((g - 1) * ax + divup(nx - g + 1, ax) * ax) * (divup(ny, ay) * ay)


_QPRIM = {}


def _quant_prim(layout, has_rope):
  """Primitive for layout in ("both", "row", "col"); outputs (row, col, row scales, col scales) restricted to it."""
  if (layout, has_rope) in _QPRIM:
    return _QPRIM[layout, has_rope]
  try:
    from jax.extend.core import Primitive  # pylint: disable=import-outside-toplevel
  except ImportError:  # pragma: no cover
    from jax.core import Primitive  # pylint: disable=import-outside-toplevel
  from jax._src import dispatch  # pylint: disable=import-outside-toplevel
  from jax.interpreters import mlir  # pylint: disable=import-outside-toplevel
  from transformer_engine.jax.triton_extensions import utils as te_triton  # pylint: disable=import-outside-toplevel
  import os  # pylint: disable=import-outside-toplevel

  # launch config: DSV4_OMX_CFG="slabs,warps" (slabs of 32 tokens per program); swept on GB300 (see REPORT)
  default = {"rope_tiles": "1,2", "norope_tiles": "1,2"}.get(layout, "1,1" if has_rope else "1,2")
  env = os.environ.get(f"DSV4_OMX_CFG_{layout.upper()}", os.environ.get("DSV4_OMX_CFG", default))
  slabs, warps = (int(v) for v in env.split(","))
  fn = {
      "both": _quant_gm_kernel,
      "row": _quant_gm_row_kernel,
      "col": _quant_gm_col_kernel,
      "norope_tiles": _quant_gm_norope_tiles_kernel,
      "rope_tiles": _quant_gm_rope_tiles_kernel,
  }[layout]
  kernel = triton.autotune(configs=[triton.Config({}, num_warps=warps, num_stages=1)], key=[])(fn)
  prim = Primitive(f"dsv4_o_mxfp8_quant_gm_{layout}{'_rope' if has_rope else ''}")
  prim.multiple_results = True
  prim_has_rope = has_rope

  def abstract(x, cos, sin, *prev, groups, d, nope, has_rope):
    del cos, sin, d, nope, prev
    assert has_rope == prim_has_rope
    bt, c = x.shape
    kg = c // groups
    u8 = jnp.uint8
    data = jax.core.ShapedArray((groups * bt * kg,), u8)
    rs = jax.core.ShapedArray((_grouped_scale_size(groups * bt, kg, groups, False),), u8)
    cs = jax.core.ShapedArray((_grouped_scale_size(groups * bt, kg, groups, True),), u8)
    return {"row": (data, rs), "col": (data, cs)}.get(layout, (data, data, rs, cs))

  def lowering(ctx, x, cos, sin, *prev, groups, d, nope, has_rope):
    bt, c = ctx.avals_in[0].shape
    kg = c // groups
    tiles, tph = c // _TILE, d // _TILE
    if layout == "norope_tiles":
      tiles = tiles // tph * (tph - 1)
    elif layout == "rope_tiles":
      tiles = tiles // tph
    return te_triton.triton_call_lowering(
        ctx,
        kernel,
        x,
        cos,
        sin,
        *prev,
        grid=(bt // (32 * slabs), tiles),
        input_output_aliases={3: 0, 4: 1, 5: 2, 6: 3} if layout == "rope_tiles" else None,
        constexprs={
            "BT": bt,
            "G": groups,
            "KG": kg,
            "D": d,
            "NOPE": nope,
            "HAS_ROPE": has_rope,
            "SLABS": slabs,
            "IDX": tl.int32 if bt * c < 2**31 else tl.int64,
        },
    )

  prim.def_abstract_eval(abstract)
  prim.def_impl(functools.partial(dispatch.apply_primitive, prim))
  mlir.register_lowering(prim, lowering, platform="cuda")
  _QPRIM[layout, has_rope] = prim
  return prim


_TWO_LAUNCH = __import__("os").environ.get("DSV4_OMX_TWO_LAUNCH", "1") == "1"


def quant_gm(x2d, cos, sin, groups, d=0, nope=0, has_rope=False, split=False):
  """x2d [BT, G * KG] bf16 -> TE ScaledTensor2x for the group-major (G * BT, KG) operand (equal group sizes BT).

  split=True runs the rowwise and the colwise quantization as two kernels, so that XLA / JAX can drop the colwise one
  when only the rowwise operand is used (the forward pass under remat, where the wgrad residual is recomputed).
  """
  from transformer_engine.jax.quantize import ScaledTensorFactory, ScalingMode, QuantizeLayout  # pylint: disable=import-outside-toplevel

  bt, c = x2d.shape
  kg = c // groups
  if bt % _TILE or kg % _TILE:
    raise ValueError(f"quant_gm needs tokens ({bt}) and group width ({kg}) divisible by {_TILE}")
  if cos is None:
    cos = sin = jnp.zeros((1, 1), jnp.float32)
  else:
    # per-tile rotary tables: the rotary channels are the last 2 * R/2 = 64 of each D-channel head, i.e. pair slots
    # 32..63 of the 128-channel tile that holds them; slots 0..31 get cos = 1, sin = 0 (exact identity).
    if d - nope != 64 or nope % _TILE != 64:
      raise ValueError("fused RoPE quantize assumes 64 rotary channels at the end of each head (D % 128 == 0)")
    cos = jnp.concatenate([jnp.ones_like(cos), cos], axis=-1).astype(jnp.float32)
    sin = jnp.concatenate([jnp.zeros_like(sin), sin], axis=-1).astype(jnp.float32)
  kw = {"groups": groups, "d": d, "nope": nope, "has_rope": has_rope}
  if split:
    row, rs = _quant_prim("row", has_rope).bind(x2d, cos, sin, **kw)
    col, cs = _quant_prim("col", has_rope).bind(x2d, cos, sin, **kw)
  elif has_rope and _TWO_LAUNCH:
    # non-rotary tiles with the faster 2-warp config, then the rotary tiles (1 warp) into the same buffers
    outs = _quant_prim("norope_tiles", has_rope).bind(x2d, cos, sin, **kw)
    row, col, rs, cs = _quant_prim("rope_tiles", has_rope).bind(x2d, cos, sin, *outs, **kw)
  else:
    row, col, rs, cs = _quant_prim("both", has_rope).bind(x2d, cos, sin, **kw)
  e8, f8 = jnp.float8_e8m0fnu, jnp.float8_e4m3fn
  return ScaledTensorFactory.create(
      data=jax.lax.bitcast_convert_type(row, f8),
      scale_inv=jax.lax.bitcast_convert_type(rs, e8),
      colwise_data=jax.lax.bitcast_convert_type(col, f8),
      colwise_scale_inv=jax.lax.bitcast_convert_type(cs, e8),
      scaling_mode=ScalingMode.MXFP8_1D_SCALING,
      dq_dtype=x2d.dtype,
      q_layout=QuantizeLayout.ROWWISE_COLWISE,
      data_layout="NN",
      flatten_axis=1,
      first_dims=jnp.full((groups,), bt, jnp.int32),
      original_shape=(groups * bt, kg),
      pre_swizzled=True,
  )


# DSV4_OMX_SPLIT=1: separate rowwise / colwise quantize kernels for the o input (colwise dropped in the non-remat
# forward). Measured as a wash on GB300 (row 0.50 + col 0.61 ms vs both 0.80 ms per call), so off by default.
_SPLIT = __import__("os").environ.get("DSV4_OMX_SPLIT", "0") == "1"


def _fused_o_proj_local(o, cos, sin, w):
  """Per-shard o [b, s, H, D] -> inv-RoPE -> grouped MXFP8 GEMM with w [G, K, N] -> [b, s, G, N] (custom VJP).

  fwd:  one Triton kernel (inv-RoPE + transpose + MXFP8 row/col quantize) + TE grouped GEMM.
  bwd:  one Triton kernel quantizes dy into the group-major layout; TE grouped GEMMs give the dgrad (already in the
        [G, b, s, K] layout the fused o-backward kernel reads) and the wgrad; the dsv4_qo_fused o-backward kernel
        applies the transposed inverse RoPE.
  """
  import transformer_engine.jax.cpp_extensions as tex  # pylint: disable=import-outside-toplevel
  from transformer_engine.jax.quantize import TensorUsage  # pylint: disable=import-outside-toplevel

  b, s, h, d = o.shape
  g, k, n = w.shape
  bt = b * s
  rope_half = cos.shape[-1]
  nope = d - 2 * rope_half

  def fwd(o, cos, sin, w):
    xq = quant_gm(
        o.reshape(bt, h * d),
        cos.reshape(bt, rope_half),
        sin.reshape(bt, rope_half),
        g,
        d=d,
        nope=nope,
        has_rope=True,
        split=_SPLIT,
    )
    wq = tex.grouped_quantize(w.astype(o.dtype), _mx_set(g).kernel, flatten_axis=-1)
    y = tex.grouped_gemm(
        xq.get_tensor(usage=TensorUsage.LHS), wq.get_tensor(usage=TensorUsage.RHS), contracting_dims=((1,), (1,))
    )
    y = jnp.transpose(y.reshape(g, b, s, n), (1, 2, 0, 3))
    return y, (xq.get_tensor(usage=TensorUsage.LHS_TRANS), wq.get_tensor(usage=TensorUsage.RHS_TRANS), cos, sin)

  @jax.custom_vjp
  def f(o, cos, sin, w):
    return fwd(o, cos, sin, w)[0]

  def bwd(res, dy):
    x_t, w_t, cos, sin = res
    dyq = quant_gm(dy.astype(o.dtype).reshape(bt, g * n), None, None, g)
    dgrad = tex.grouped_gemm(dyq.get_tensor(usage=TensorUsage.LHS), w_t, contracting_dims=((1,), (2,)))
    wgrad = tex.grouped_gemm(x_t, dyq.get_tensor(usage=TensorUsage.RHS), contracting_dims=((0,), (0,)))
    do = _qo()._prims()["o_bwd"].bind(dgrad.reshape(g, b, s, k), cos, sin, heads=h, nope=nope)  # pylint: disable=protected-access
    return do, jnp.zeros_like(cos), jnp.zeros_like(sin), wgrad.astype(w.dtype)

  f.defvjp(fwd, bwd)
  return f(o, cos, sin, w)


def inv_rope_grouped_proj_mxfp8_fused(o, cos, sin, w):
  """Fused-kernel MXFP8 version of dsv4_qo_fused.inv_rope_grouped_proj: o [B, S, H, D], w [G, I, O] -> [B, S, G, O]."""
  qo = _qo()
  ax = qo._leading_batch_axis()  # pylint: disable=protected-access
  spec4, spec3 = P(ax, None, None, None), P(ax, None, None)
  return qo._shard(_fused_o_proj_local, (spec4, spec3, spec3, P(None, None, None)), spec4)(  # pylint: disable=protected-access
      o, cos.astype(jnp.float32), sin.astype(jnp.float32), w
  )
