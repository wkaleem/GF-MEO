"""Bounds on demand inputs from the GENCO data-generation ranges."""

import numpy as np


def virtual_load_box(ratios, radius, *, global_range, local_range):
    r = np.asarray(ratios, dtype=float)
    g = np.asarray(global_range, dtype=float)
    s = np.asarray(local_range, dtype=float)
    if r.ndim != 1 or not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("Load ratios must be finite, positive and one-dimensional")
    if not np.isfinite(radius) or not 0 <= radius < 1:
        raise ValueError("Fractional radius must lie in [0,1)")
    if any(
        a.shape != (2,) or not np.isfinite(a).all() or not 0 < a[0] <= a[1]
        for a in (g, s)
    ):
        raise ValueError("Positive ordered factor ranges required")
    center_lo = max(float(g[0]), float(np.max(r / s[1], initial=-np.inf)))
    center_hi = min(float(g[1]), float(np.min(r / s[0], initial=np.inf)))
    lo, hi = center_lo, center_hi
    roundoff = 32 * np.finfo(float).eps * (1 + max(abs(center_lo), abs(center_hi)))
    shifted = center_lo > center_hi + roundoff
    if shifted:
        # Every virtual coordinate must belong to both r_i*[1-R,1+R]
        # and alpha*[s_min,s_max]. Eliminating the coordinates gives this
        # necessary and sufficient common-alpha interval.
        lo = max(float(g[0]), float(np.max((1 - radius) * r / s[1], initial=-np.inf)))
        hi = min(float(g[1]), float(np.min((1 + radius) * r / s[0], initial=np.inf)))
    roundoff = 32 * np.finfo(float).eps * (1 + max(abs(lo), abs(hi)))
    if lo > hi + roundoff:
        raise ValueError(
            "Empty virtual-load box within the prescribed radius and factor ranges"
        )
    alpha = float(np.clip((lo + hi) / 2, g[0], g[1]))
    lower = np.maximum(-radius, s[0] * alpha / r - 1)
    upper = np.minimum(radius, s[1] * alpha / r - 1)
    reversed_interval = lower > upper
    if (
        (lower - upper)
        > 32 * np.finfo(float).eps * (1 + np.maximum(np.abs(lower), np.abs(upper)))
    ).any():
        raise ValueError("Numerically empty virtual-load coordinate interval")
    if reversed_interval.any():
        point = np.clip((lower + upper) / 2, -radius, radius)
        lower = np.where(reversed_interval, point, lower)
        upper = np.where(reversed_interval, point, upper)
    return (
        lower,
        upper,
        dict(
            fixed_common_load_scale=alpha,
            admissible_scale_interval=[lo, hi],
            physical_center_scale_interval=[center_lo, center_hi],
            physical_center_admissible=not shifted,
            shifted_input_box=shifted,
            policy="Preserve center-containing boxes when available; otherwise intersect virtual adjustments with the fixed factor ranges.",
        ),
    )
