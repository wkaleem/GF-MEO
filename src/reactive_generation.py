"""Allocation of bus reactive generation among online generators."""

import numpy as np


def allocation_coefficients(bus, online, lower, upper):
    bus = np.asarray(bus)
    online = np.asarray(online, float)
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    if (
        bus.ndim != 1
        or any(a.shape != bus.shape for a in (online, lower, upper))
        or not np.isfinite(bus).all()
        or np.any(bus < 0)
        or np.any(bus != bus.astype(int))
        or not np.isin(online, [0.0, 1.0]).all()
        or not np.isfinite(np.r_[lower, upper]).all()
        or np.any(lower > upper)
    ):
        raise ValueError("Invalid generator incidence, status or reactive bounds")
    weights = np.zeros(len(bus))
    offset = np.zeros(len(bus))
    for index in np.unique(bus):
        active = np.flatnonzero((bus == index) & (online == 1))
        if not len(active):
            continue
        if len(active) == 1:
            weights[active] = 1.0
            continue  # Preserve the original single-unit output exactly.
        widths = upper[active] - lower[active]
        total = float(widths.sum())
        weights[active] = widths / total if total > 0 else 1.0 / len(active)
        offset[active] = lower[active] - weights[active] * lower[active].sum()
    return weights, offset
