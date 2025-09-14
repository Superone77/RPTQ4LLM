import torch, triton, triton.language as tl

FP4_MAX = 6.0         # fp4-(1-2-1) 动态范围上界

# ---------------------------------------------------------------
# Triton kernel：MXFP4 量化，只使用 per-block scaling
# ---------------------------------------------------------------
@triton.jit
def mxfp4_fwd_kernel(
    x_ptr,          # *f16 / *f32  | 输入
    out_ptr,        # *f16 / *f32  | 输出
    prob_ptr,       # *f32         | [0,1) 随机数 (or 全 0)
    M,              # int32        | 总元素数
    BLOCK: tl.constexpr,          # =32
    STOCHASTIC: tl.constexpr,     # bool
):
    pid   = tl.program_id(0)
    offs  = pid * BLOCK + tl.arange(0, BLOCK)
    mask  = offs < M

    # -------- 读入数据 --------
    x      = tl.load(x_ptr + offs,  mask=mask, other=0.0)
    prob   = tl.load(prob_ptr + offs, mask=mask, other=0.0)
    sign   = tl.where(x >= 0.0, 1.0, -1.0)
    x_abs  = tl.abs(x)

    # -------- 计算 per-block scale_b --------
    blk_max = tl.max(tl.where(mask, x_abs, 0.0), axis=0)
    # 避免除零 / inf
    scale_b = tl.where(blk_max > 0.0, 6.0 / blk_max, 1.0)

    y = x_abs * scale_b                # bring into [0, 6]

    # ============================================================
    # 1-2-1 fp4 量化 (MXFP4 格式)
    # ============================================================
    if STOCHASTIC:
        # 上、下邻格
        hi = tl.where(
            y > 4, 6.0,
            tl.where(y > 3, 4.0,
                tl.where(y > 2, 3.0,
                    tl.where(y > 1.5, 2.0,
                        tl.where(y > 1.0, 1.5,
                            tl.where(y > 0.5, 1.0, 0.5)))))
        )
        lo = tl.where(
            y > 4, 4.0,
            tl.where(y > 3, 3.0,
                tl.where(y > 2, 2.0,
                    tl.where(y > 1.5, 1.5,
                        tl.where(y > 1.0, 1.0,
                            tl.where(y > 0.5, 0.5, 0.0)))))
        )
        prob_up = (y - lo) / (hi - lo + 1e-7)         # 1e-7 防除零
        q_abs_blk = tl.where(prob < prob_up, hi, lo)
    else:
        # 确定性 (阈值中心点)
        q_abs_blk = tl.where(
            y > 5,   6.0,
            tl.where(y > 3.5, 4.0,
                tl.where(y > 2.5, 3.0,
                    tl.where(y > 1.75, 2.0,
                        tl.where(y > 1.25, 1.5,
                            tl.where(y > 0.75, 1.0,
                                tl.where(y > 0.25, 0.5, 0.0))))))
        )

    # -------- 反缩放并写回 --------
    q_abs = q_abs_blk / scale_b
    q_val = sign * q_abs
    tl.store(out_ptr + offs, q_val, mask=mask)


# ---------------------------------------------------------------
# Python 包装：MXFP4 量化
# ---------------------------------------------------------------
def mxfp4_forward(
    x: torch.Tensor,
    stochastic_rounding: bool = False,
):
    """
    x             : CUDA 张量 (f16/f32)，任意形状
    stochastic_rounding: 是否使用随机舍入
    """
    
    assert x.is_cuda and x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    fp_dtype = x.dtype
    orig_shape = x.shape
    x_flat     = x.contiguous().view(-1)
    M          = x_flat.numel()

    # ---- 随机概率张量 ----
    if stochastic_rounding:
        prob = torch.rand_like(x_flat, dtype=torch.float32)
    else:
        prob = torch.zeros_like(x_flat, dtype=torch.float32)

    out   = torch.empty_like(x_flat)
    BLOCK = 32  # MXFP4使用32的block size
    grid  = ((M + BLOCK - 1) // BLOCK,)

    mxfp4_fwd_kernel[grid](
        x_flat, out, prob, M,
        BLOCK=BLOCK,
        STOCHASTIC=stochastic_rounding,
        num_warps=4
    )
    return out.view(orig_shape).to(fp_dtype)


# ---------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 32
    x = torch.randn(B, N, device="cuda", dtype=torch.float16) * 5

    y_det = mxfp4_forward(x, stochastic_rounding=False)
    y_sto = mxfp4_forward(x, stochastic_rounding=True)
    print("MXFP4 det  :", y_det[0, :8])
    print("MXFP4 sto  :", y_sto[0, :8])
