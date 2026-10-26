"""Shared augmented-Lagrangian objectives for horizon and period solves."""

import numpy as np


def period_objective(interface, target, multipliers, rho, weights):
    delta = np.asarray(interface) - np.asarray(target)
    return float(
        np.dot(multipliers, interface) + rho / 2 * np.dot(weights, delta * delta)
    )


def horizon_objective(cost, interface, target, multipliers, rho, weights):
    delta = np.asarray(target) - np.asarray(interface)
    return float(
        cost
        - np.sum(multipliers * interface)
        + rho / 2 * np.sum(weights * delta * delta)
    )
