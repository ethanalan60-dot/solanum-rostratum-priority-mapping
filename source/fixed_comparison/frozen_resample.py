"""Cell-overlap averaging on the same rectangular footprint.

Unlike adaptive pooling, arbitrary source/target grids use exact pixel-boundary
overlap weights. This utility is for future support-grid experiments and is NOT
used by any existing local_v2 run. It does not model an optical point-spread function.
"""
from functools import lru_cache
import torch


@lru_cache(maxsize=64)
def overlap_weights(n_source, n_target, device, dtype):
    if n_source <= 0 or n_target <= 0:
        raise ValueError("Grid dimensions must be positive")
    # Build in float64 before casting; rows integrate one output cell.
    source = torch.arange(n_source + 1, dtype=torch.float64, device=device)
    target = torch.arange(n_target + 1, dtype=torch.float64, device=device) * (n_source / n_target)
    overlap = (torch.minimum(target[1:, None], source[None, 1:])
               - torch.maximum(target[:-1, None], source[None, :-1])).clamp_min(0)
    return (overlap / (n_source / n_target)).to(dtype=dtype)


def area_resize(x, size):
    """BCHW floating tensor -> BCH'W', cell averages on equal-area output cells."""
    if x.ndim != 4 or not x.is_floating_point():
        raise ValueError("Expected a floating BCHW tensor")
    height, width = map(int, size)
    if height <= 0 or width <= 0:
        raise ValueError("Grid dimensions must be positive")
    # Keep scientific area calculations out of mixed-precision matmul.
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        x = x.to(dtype)
        wy = overlap_weights(x.shape[-2], height, str(x.device), dtype)
        wx = overlap_weights(x.shape[-1], width, str(x.device), dtype)
        return torch.matmul(torch.matmul(wy, x), wx.transpose(0, 1))


def normalized_area_resize(values, valid, size):
    """Return valid-domain conditional values and output valid-area fractions.

    valid is B1HW in [0,1]. Coverage is sum(values*valid)/sum(valid),
    with a zero result on an empty domain. Same-footprint averages conserve it.
    """
    if valid.ndim != 4 or valid.shape[1] != 1 or values.shape[0] != valid.shape[0] or values.shape[-2:] != valid.shape[-2:]:
        raise ValueError("Expected values BCHW and valid B1HW with matching grid")
    if not values.is_floating_point():
        raise ValueError("Values must be floating point")
    dtype = torch.float64 if values.dtype == torch.float64 or valid.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=values.device.type, enabled=False):
        # Promote BEFORE multiplication so fractional half-precision valid mass cannot underflow.
        values, valid = values.to(dtype), valid.to(dtype)
        if not torch.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any():
            raise ValueError("Valid-area fractions must be finite and in [0, 1]")
        # NaN*0 is NaN. Ignore unspecified values outside the valid domain explicitly.
        clean_values = torch.where(valid > 0, values, torch.zeros_like(values))
        if not torch.isfinite(clean_values).all():
            raise ValueError("Values must be finite wherever valid is positive")
        weights = area_resize(valid, size)
        numerator = area_resize(clean_values * valid, size)
        denominator = torch.where(weights > 0, weights, torch.ones_like(weights))
        result = torch.where(weights > 0, numerator / denominator, torch.zeros_like(numerator))
    return result, weights
