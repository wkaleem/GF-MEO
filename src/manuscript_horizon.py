"""UC horizon constraints, generation costs and dispatch consensus.

Interface powers are in per unit on the network base; costs are unscaled.
"""

from __future__ import annotations
import time
import numpy as np
from pyscipopt import Model, quicksum

SPEC = "manuscript_20261024_literal_uc_and_initial_dispatch_ramps"


def arrays(case):
    T, G = int(case["T"]), len(case["gen"])
    base = float(case["baseMVA"])
    if T < 1 or T != case["T"] or G < 1 or not np.isfinite(base) or base <= 0:
        raise ValueError("Invalid horizon, generator count, or power base")
    for row in case["gen"]:
        for key in [
            "min_p_mw",
            "max_p_mw",
            "cp0_eur",
            "cp1_eur_per_mw",
            "cp2_eur_per_mw2",
        ]:
            if not np.isfinite(row[key]):
                raise ValueError(key)
        if row["min_p_mw"] > row["max_p_mw"] or row["max_p_mw"] < 0:
            raise ValueError("Invalid generation bounds")

    def vector(key):
        a = np.asarray(case[key], float)
        if a.shape != (G,) or not np.isfinite(a).all():
            raise ValueError(key)
        return a

    data = {
        k: vector(k)
        for k in [
            "initial_on",
            "initial_pg_mw",
            "ramp_up_mw",
            "ramp_down_mw",
            "startup_ramp_mw",
            "shutdown_ramp_mw",
            "min_up",
            "min_down",
            "initial_forced_on",
            "initial_forced_off",
            "startup_cost",
            "shutdown_cost",
            "on_cost",
        ]
    }
    for name in ["min_up", "min_down", "initial_forced_on", "initial_forced_off"]:
        a = data[name]
        if (a < 0).any() or (a != a.astype(int)).any():
            raise ValueError(name)
    if (data["min_up"] < 1).any() or (data["min_down"] < 1).any():
        raise ValueError("Minimum durations must be positive integers")
    if not np.isin(data["initial_on"], [0, 1]).all():
        raise ValueError("initial_on")
    if any((data[k] > T).any() for k in ["initial_forced_on", "initial_forced_off"]):
        raise ValueError(
            "Initial obligation indices must lie within the declared horizon"
        )
    if any(
        (data[k] < 0).any()
        for k in ["ramp_up_mw", "ramp_down_mw", "startup_ramp_mw", "shutdown_ramp_mw"]
    ):
        raise ValueError("Negative ramp limit")
    data["pmin"] = np.maximum(0.0, [g["min_p_mw"] / base for g in case["gen"]])
    data["pmax"] = np.asarray([g["max_p_mw"] / base for g in case["gen"]])
    return T, G, base, data


def check_horizon(case, solution, tolerance=1e-5):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Invalid feasibility tolerance")
    T, G, base, a = arrays(case)
    values = {k: np.asarray(solution[k], float) for k in ["u", "su", "sd", "p"]}
    if any(v.shape != (T, G) or not np.isfinite(v).all() for v in values.values()):
        return {"accepted": False, "reason": "invalid shape or nonfinite values"}
    u, su, sd, p = (values[k] for k in ["u", "su", "sd", "p"])
    discrete = all(np.isin(values[k], [0, 1]).all() for k in ["u", "su", "sd"])
    checks = {}

    def add(name, value, equality=False):
        b = np.asarray(value, float)
        checks[name] = float(
            np.max(np.abs(b) if equality else np.maximum(b, 0), initial=0)
        )

    add("p_lower", a["pmin"] * u - p)
    add("p_upper", p - a["pmax"] * u)
    add("transition_t2_onward", u[:-1] - u[1:] + su[1:] - sd[1:], True)
    initial = []
    up = []
    down = []
    for g in range(G):
        initial.extend(1 - u[: int(a["initial_forced_on"][g]), g])
        initial.extend(u[: int(a["initial_forced_off"][g]), g])
        U, D = int(a["min_up"][g]), int(a["min_down"][g])
        if U < 1 or D < 1:
            raise ValueError("Pilot requires positive minimum durations")
        for t in range(U - 1, T):
            up.append(su[t - U + 1 : t + 1, g].sum() - u[t, g])
        for t in range(D - 1, T):
            down.append(sd[t - D + 1 : t + 1, g].sum() - 1 + u[t, g])
    add("initial_obligations", initial)
    add("minimum_up", up)
    add("minimum_down", down)
    prev_p = np.vstack((a["initial_pg_mw"] / base, p[:-1]))
    prev_u = np.vstack((a["initial_on"], u[:-1]))
    add(
        "ramp_up",
        p - prev_p - (a["ramp_up_mw"] * prev_u + a["startup_ramp_mw"] * su) / base,
    )
    add(
        "ramp_down",
        prev_p - p - (a["ramp_down_mw"] * u + a["shutdown_ramp_mw"] * sd) / base,
    )
    return {
        "accepted": bool(discrete and max(checks.values(), default=0) <= tolerance),
        "exact_binary": bool(discrete),
        "max_violations": checks,
        "tolerance": tolerance,
        "specification": SPEC,
    }


def economic_cost(case, solution):
    T, G, base, a = arrays(case)
    p = np.asarray(solution["p"]) * base
    c0, c1, c2 = [
        np.asarray([g[k] for g in case["gen"]])
        for k in ["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]
    ]
    return float(
        np.sum(
            c0
            + c1 * p
            + c2 * p * p
            + a["on_cost"] * solution["u"]
            + a["startup_cost"] * solution["su"]
            + a["shutdown_cost"] * solution["sd"]
        )
    )


def solve_horizon(
    case,
    *,
    target=None,
    multipliers=None,
    rho=0.0,
    weights=None,
    fixed_u=None,
    fixed_p=None,
    time_limit=20.0,
    previous=None,
):
    started = time.monotonic()
    T, G, base, a = arrays(case)
    if not np.isfinite(time_limit) or time_limit <= 0:
        raise ValueError("Invalid horizon time budget")
    target = np.zeros((T, 2 * G)) if target is None else np.asarray(target, float)
    lam = (
        np.zeros((T, 2 * G)) if multipliers is None else np.asarray(multipliers, float)
    )
    w = np.ones(2 * G) if weights is None else np.asarray(weights, float)
    if target.shape != (T, 2 * G) or lam.shape != target.shape or w.shape != (2 * G,):
        raise ValueError("Interface shape")
    if (
        not all(np.isfinite(v).all() for v in [target, lam, w])
        or (w <= 0).any()
        or not np.isfinite(rho)
        or rho < 0
    ):
        raise ValueError("AL parameters")
    for value in [fixed_u, fixed_p]:
        if value is not None and (
            np.asarray(value).shape != (T, G) or not np.isfinite(value).all()
        ):
            raise ValueError("Invalid completion data")
    if fixed_u is not None and not np.isin(fixed_u, [0, 1]).all():
        raise ValueError("Nonbinary completion pattern")
    model = Model("manuscript UC horizon")
    model.hideOutput()
    model.setRealParam("limits/time", time_limit)
    model.setRealParam("numerics/feastol", 1e-8)
    model.setRealParam("limits/gap", 1e-6)
    model.setIntParam("parallel/maxnthreads", 1)
    u, su, sd, p = ({} for _ in range(4))
    for t in range(T):
        for g in range(G):
            u[t, g] = model.addVar(vtype="B", name=f"u_{t}_{g}")
            su[t, g] = model.addVar(vtype="B", name=f"su_{t}_{g}")
            sd[t, g] = model.addVar(vtype="B", name=f"sd_{t}_{g}")
            p[t, g] = model.addVar(lb=0, ub=float(a["pmax"][g]), name=f"p_{t}_{g}")
            model.addCons(p[t, g] >= float(a["pmin"][g]) * u[t, g])
            model.addCons(p[t, g] <= float(a["pmax"][g]) * u[t, g])
            if t < int(a["initial_forced_on"][g]):
                model.addCons(u[t, g] == 1)
            if t < int(a["initial_forced_off"][g]):
                model.addCons(u[t, g] == 0)
            if t:
                model.addCons(u[t - 1, g] - u[t, g] + su[t, g] - sd[t, g] == 0)
            prev_p = p[t - 1, g] if t else float(a["initial_pg_mw"][g] / base)
            prev_u = u[t - 1, g] if t else float(a["initial_on"][g])
            model.addCons(
                p[t, g] - prev_p
                <= (
                    float(a["ramp_up_mw"][g]) * prev_u
                    + float(a["startup_ramp_mw"][g]) * su[t, g]
                )
                / base
            )
            model.addCons(
                prev_p - p[t, g]
                <= (
                    float(a["ramp_down_mw"][g]) * u[t, g]
                    + float(a["shutdown_ramp_mw"][g]) * sd[t, g]
                )
                / base
            )
            if fixed_u is not None:
                model.addCons(u[t, g] == float(fixed_u[t, g]))
            if fixed_p is not None:
                model.addCons(p[t, g] == float(fixed_p[t, g]))
    for g in range(G):
        U, D = int(a["min_up"][g]), int(a["min_down"][g])
        for t in range(U - 1, T):
            model.addCons(
                quicksum(su[i, g] for i in range(t - U + 1, t + 1)) <= u[t, g]
            )
        for t in range(D - 1, T):
            model.addCons(
                quicksum(sd[i, g] for i in range(t - D + 1, t + 1)) <= 1 - u[t, g]
            )
    terms = []
    for t in range(T):
        for g, row in enumerate(case["gen"]):
            terms.append(
                row["cp0_eur"]
                + row["cp1_eur_per_mw"] * base * p[t, g]
                + row["cp2_eur_per_mw2"] * (base * p[t, g]) ** 2
                + float(a["on_cost"][g]) * u[t, g]
                + float(a["startup_cost"][g]) * su[t, g]
                + float(a["shutdown_cost"][g]) * sd[t, g]
            )
            for index, var in [(g, u[t, g]), (G + g, p[t, g])]:
                terms.append(
                    -float(lam[t, index]) * var
                    + rho / 2 * float(w[index]) * (float(target[t, index]) - var) ** 2
                )
    epi = model.addVar(lb=-1e20, name="objective_epigraph")
    model.addCons(epi >= quicksum(terms))
    model.setObjective(epi, "minimize")
    # The incumbent is retained explicitly by the coordinator, regardless of
    # whether the solver's numerical warm-start mechanism accepts it.
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0:
        return {
            "solver_status": "budget_exhausted_during_construction",
            "elapsed_s": time.monotonic() - started,
            "specification": SPEC,
            "solution": None,
        }
    model.setRealParam("limits/time", remaining)
    model.optimize()
    result = {
        "solver_status": str(model.getStatus()),
        "elapsed_s": time.monotonic() - started,
        "bound_scope": "augmented horizon only; not an original UC-ACOPF bound",
        "specification": SPEC,
        "solution": None,
    }
    if model.getNSols():
        sol = {
            k: np.asarray(
                [
                    [
                        (
                            round(model.getVal(d[t, g]))
                            if k != "p"
                            else model.getVal(d[t, g])
                        )
                        for g in range(G)
                    ]
                    for t in range(T)
                ]
            )
            for k, d in [("u", u), ("su", su), ("sd", sd), ("p", p)]
        }
        # Preserve exact prescribed values in completion, then independently
        # check every constraint using them rather than a solver copy.
        if fixed_u is not None:
            sol["u"] = np.asarray(fixed_u).copy()
        if fixed_p is not None:
            sol["p"] = np.asarray(fixed_p).copy()
        check = check_horizon(case, sol)
        result["check"] = check
        if check["accepted"]:
            result["solution"] = sol
            result["original_cost"] = economic_cost(case, sol)
    return result
