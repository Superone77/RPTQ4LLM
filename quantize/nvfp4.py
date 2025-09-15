import os
import torch

FP8_E4M3_MAX = 448.0


def fp4_121_positive(x: torch.Tensor, stochastic_rounding: bool = False) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)

    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)


def quant_nvfp4(
    x: torch.Tensor,
    stochastic_rounding: bool = False,
    batch_size: int = 1,
    vari_length: bool = False,
) -> torch.Tensor:
    quant_mode = os.environ.get("QUANT_MODE", "")
    fp4_121_max = 6.0
    ori_shape = x.shape
    x = x.reshape(-1, 16)
    sign = x.sign()
    x_abs = x.abs()
    nvfp4_max = fp4_121_max * FP8_E4M3_MAX
    scale_per_t = x_abs.max() / nvfp4_max
    if quant_mode in ["Dynamic_Block", "Calib_Block"]:
        scale_per_t = torch.tensor(1.0, device=x.device, dtype=x.dtype)
    x_abs_scaled = x_abs / scale_per_t

    if batch_size == 1:
        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
    else:
        scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]

    input_tensor = fp4_121_max / scale_per_b
    down_cast = input_tensor.to(torch.float8_e4m3fn)
    up_cast = down_cast.to(scale_per_b.dtype)
    scale_per_b = torch.where(
        (0 < up_cast) & (up_cast < torch.inf),
        up_cast,
        torch.tensor(1.0, device=scale_per_b.device, dtype=scale_per_b.dtype),
    )

    x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

    return (sign * x_fp4_abs * scale_per_t).reshape(ori_shape)
