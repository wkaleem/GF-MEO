"""AC power flow equations and operating limits."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def _array(value, shape, name):
    result = np.asarray(value, dtype=float)
    if result.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name}: nonfinite value(s)")
    return result


def _positive_max(array):
    return float(np.maximum(np.asarray(array), 0.0).max(initial=0.0))


def _maximum_abs(array):
    return float(np.abs(array).max(initial=0.0))


def _worst(array):
    values = np.asarray(array)
    if not values.size:
        return {"index": None, "signed_value": 0.0}
    index = np.unravel_index(np.abs(values).argmax(), values.shape)
    return {"index": [int(i) for i in index], "signed_value": float(values[index])}


def ac_quantities(case, vm, va):
    """Return both-end complex branch powers and bus outgoing power, in MVA.

    A pi branch with complex tap a has currents
    If = ((ys + j*b/2)/|a|^2)*Vf - (ys/conj(a))*Vt,
    It = ((ys + j*b/2))*Vt - (ys/a)*Vf.
    Shunt consumption is base_shunt*(GS-j*BS)*|V|^2.
    """
    buses, branches = case["bus"], case["branch"]
    nb, nl = len(buses), len(branches)
    voltage_magnitudes = np.asarray(vm, dtype=float)
    if voltage_magnitudes.ndim != 2 or voltage_magnitudes.shape[1] != nb:
        raise ValueError("vm must be a time-by-bus matrix")
    nt = voltage_magnitudes.shape[0]
    voltage_magnitudes = _array(vm, (nt, nb), "vm")
    angles = _array(va, (nt, nb), "va")
    bus_index = {int(bus["bus"]): i for i, bus in enumerate(buses)}
    if len(bus_index) != nb:
        raise ValueError("Duplicate bus IDs")
    base = float(case["baseMVA"])
    if not np.isfinite(base) or base <= 0:
        raise ValueError("Invalid baseMVA")
    v = voltage_magnitudes * np.exp(1j * angles)
    sf, st = np.zeros((nt, nl), complex), np.zeros((nt, nl), complex)
    outgoing = np.zeros((nt, nb), complex)
    angle_difference = np.zeros((nt, nl))
    active = np.zeros(nl, bool)
    for line, branch in enumerate(branches):
        f, t = bus_index[int(branch["from_bus"])], bus_index[int(branch["to_bus"])]
        angle_difference[:, line] = angles[:, f] - angles[:, t]
        status = float(branch.get("br_status", 1))
        if status not in (0.0, 1.0):
            raise ValueError("Branch status must be binary")
        if status == 0:
            continue
        active[line] = True
        z = complex(float(branch["r"]), float(branch["x"]))
        if z == 0:
            raise ValueError("In-service zero-impedance branch is unsupported")
        ys, charging = 1.0 / z, 0.5j * float(branch["b"])
        ratio = float(branch["tap"])
        if ratio == 0:
            ratio = 1.0  # MATPOWER's zero tap means an ordinary line.
        if ratio < 0:
            raise ValueError("Negative tap ratio")
        a = ratio * np.exp(1j * np.deg2rad(float(branch.get("shift", 0))))
        current_f = (ys + charging) / (abs(a) ** 2) * v[:, f] - ys / np.conj(a) * v[
            :, t
        ]
        current_t = (ys + charging) * v[:, t] - ys / a * v[:, f]
        sf[:, line] = base * v[:, f] * np.conj(current_f)
        st[:, line] = base * v[:, t] * np.conj(current_t)
        outgoing[:, f] += sf[:, line]
        outgoing[:, t] += st[:, line]
    shunt_base = float(case.get("shunt_baseMVA", 100.0))
    shunt = shunt_base * np.asarray([complex(b["GS"], -b["BS"]) for b in buses])
    outgoing += voltage_magnitudes**2 * shunt[None, :]
    return {
        "from_mva": sf,
        "to_mva": st,
        "outgoing_mva": outgoing,
        "angle_difference_rad": angle_difference,
        "active": active,
    }


def check_solution(
    case,
    solution,
    *,
    power_tolerance=1e-3,
    voltage_tolerance=1e-5,
    binary_tolerance=1e-6,
    angle_tolerance=1e-6,
):
    """Check the declared original AC+UC constraints; return physical residuals.

    Malformed/nonfinite solutions raise ValueError, never pass by NaN comparison.
    Cases with pre-horizon dispatch ramps are checked by feasibility.py.
    Generator constant costs are reported
    with and without commitment scaling. Minimum durations are truncated at
    the horizon.
    """
    if any(
        t <= 0 or not np.isfinite(t)
        for t in (power_tolerance, voltage_tolerance, binary_tolerance, angle_tolerance)
    ):
        raise ValueError("Tolerances must be finite and positive")
    nt, nb, ng = int(case["T"]), len(case["bus"]), len(case["gen"])
    if nt < 1:
        raise ValueError("At least one period is required")
    if case.get("initial_pg_ramp_enforced", False):
        raise ValueError(
            "This checker requires an explicit initial-Pg extension for t=0 ramping"
        )
    if "solution" in solution:
        solution = solution["solution"]
    aliases = {"pg": "pg_mw", "qg": "qg_mvar", "vm": "vm_pu", "va": "va_rad"}
    value = {
        key: solution[key] if key in solution else solution[alias]
        for key, alias in aliases.items()
    }
    pg = _array(value["pg"], (nt, ng), "pg")
    qg = _array(value["qg"], (nt, ng), "qg")
    vm = _array(value["vm"], (nt, nb), "vm")
    va = _array(value["va"], (nt, nb), "va")
    uc_entry = solution["uc"]
    u = _array(
        uc_entry["on"] if isinstance(uc_entry, dict) else uc_entry, (nt, ng), "uc"
    )
    initial = _array(case["initial_on"], (ng,), "initial_on")
    if np.any((initial != 0) & (initial != 1)):
        raise ValueError("initial_on must be binary")
    pd = _array(case["pd"], (nt, nb), "pd")
    qd = _array(case["qd"], (nt, nb), "qd")
    previous_u = np.vstack((initial, u[:-1]))
    inferred_startup = np.maximum(u - previous_u, 0)
    inferred_shutdown = np.maximum(previous_u - u, 0)
    startup_raw = solution.get(
        "startup", uc_entry.get("startup") if isinstance(uc_entry, dict) else None
    )
    shutdown_raw = solution.get(
        "shutdown", uc_entry.get("shutdown") if isinstance(uc_entry, dict) else None
    )
    startup = (
        inferred_startup
        if startup_raw is None
        else _array(startup_raw, (nt, ng), "startup")
    )
    shutdown = (
        inferred_shutdown
        if shutdown_raw is None
        else _array(shutdown_raw, (nt, ng), "shutdown")
    )
    uc_all = np.concatenate((u.ravel(), startup.ravel(), shutdown.ravel()))

    physics = ac_quantities(case, vm, va)
    buses, generators, branches = case["bus"], case["gen"], case["branch"]
    bus_index = {int(bus["bus"]): i for i, bus in enumerate(buses)}
    # The shared formulation declares reference angle zero. Do not silently
    # rotate a solver state: gauge canonicalization must be a separate action.
    reference_indices = [i for i, bus in enumerate(buses) if bool(bus.get("REF", 0))]
    injection = np.zeros((nt, nb), complex)
    for g, generator in enumerate(generators):
        injection[:, bus_index[int(generator["bus"])]] += pg[:, g] + 1j * qg[:, g]
    mismatch = injection - (pd + 1j * qd) - physics["outgoing_mva"]
    pmin, pmax, qmin, qmax = [
        np.asarray([g[key] for g in generators], float)
        for key in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar")
    ]
    vmin, vmax = [
        np.asarray([b[key] for b in buses], float) for key in ("min_vm_pu", "max_vm_pu")
    ]
    rate = np.asarray([b["rate_a"] for b in branches], float)
    limited = physics["active"] & (rate > 0)
    thermal_excess = (
        np.maximum(
            np.abs(physics["from_mva"][:, limited]),
            np.abs(physics["to_mva"][:, limited]),
        )
        - rate[limited]
    )
    angle_excess = []
    for line, branch in enumerate(branches):
        if not physics["active"][line]:
            continue
        delta = physics["angle_difference_rad"][:, line]
        lower, upper = float(branch.get("ang_min", -360)), float(
            branch.get("ang_max", 360)
        )
        if lower > -360:
            angle_excess.extend(np.deg2rad(lower) - delta)
        if upper < 360:
            angle_excess.extend(delta - np.deg2rad(upper))

    ramp = (
        _array(case["ramp_mw"], (ng,), "ramp_mw")
        if "ramp_mw" in case
        else pmax * float(case["ramp_ratio"])
    )
    if np.any(ramp < 0) or not np.all(np.isfinite(ramp)):
        raise ValueError("Ramp limits must be finite and nonnegative")
    dp = pg[1:] - pg[:-1]
    ramp_up = dp - ramp * (u[:-1] + startup[1:])
    ramp_down = -dp - ramp * (u[1:] + shutdown[1:])
    fixed_values = np.asarray(case.get("fixed_on", []), dtype=float)
    if (
        fixed_values.ndim != 1
        or not np.isfinite(fixed_values).all()
        or np.any(fixed_values != np.floor(fixed_values))
    ):
        raise ValueError("fixed_on must contain finite integer indices")
    fixed = fixed_values.astype(int)
    if np.any((fixed < 0) | (fixed >= ng)):
        raise ValueError("fixed_on index outside generator array")
    forced_on = np.asarray(case.get("initial_forced_on", [0] * ng), dtype=int)
    forced_off = np.asarray(case.get("initial_forced_off", [0] * ng), dtype=int)
    min_up = np.asarray(case["min_up"], dtype=int)
    min_down = np.asarray(case["min_down"], dtype=int)
    for key, values in (
        ("initial_forced_on", forced_on),
        ("initial_forced_off", forced_off),
        ("min_up", min_up),
        ("min_down", min_down),
    ):
        if values.shape != (ng,) or np.any(values < 0):
            raise ValueError(f"Invalid {key}")
        original = np.asarray(case.get(key, values), float)
        if np.any(original != values):
            raise ValueError(f"{key} must contain integer periods")
    if np.any((forced_on > 0) & (initial == 0)) or np.any(
        (forced_off > 0) & (initial == 1)
    ):
        raise ValueError("Initial forced status contradicts initial_on")
    initial_excess, up_excess, down_excess = [], [], []
    for g in range(ng):
        initial_excess.extend(1 - u[: min(nt, forced_on[g]), g])
        initial_excess.extend(u[: min(nt, forced_off[g]), g])
        for t in range(nt):
            # Pairwise equivalent of truncated dwell requirements for binary schedules.
            up_excess.extend(startup[t, g] - u[t : min(nt, t + min_up[g]), g])
            down_excess.extend(
                shutdown[t, g] - (1 - u[t : min(nt, t + min_down[g]), g])
            )

    maxima = {
        "active_balance_mw": _maximum_abs(mismatch.real),
        "reactive_balance_mvar": _maximum_abs(mismatch.imag),
        "thermal_mva": _positive_max(thermal_excess),
        "voltage_pu": max(_positive_max(vmin - vm), _positive_max(vm - vmax)),
        "active_generation_mw": max(
            _positive_max(u * pmin - pg), _positive_max(pg - u * pmax)
        ),
        "reactive_generation_mvar": max(
            _positive_max(u * qmin - qg), _positive_max(qg - u * qmax)
        ),
        "angle_difference_rad": _positive_max(angle_excess),
        "reference_angle_rad": _maximum_abs(va[:, reference_indices]),
        "ramp_up_mw": _positive_max(ramp_up),
        "ramp_down_mw": _positive_max(ramp_down),
        "integrality": _maximum_abs(uc_all - np.rint(uc_all)),
        "binary_bounds": max(_positive_max(-uc_all), _positive_max(uc_all - 1)),
        "uc_transition": _maximum_abs(u - previous_u - startup + shutdown),
        "startup_definition": _maximum_abs(startup - inferred_startup),
        "shutdown_definition": _maximum_abs(shutdown - inferred_shutdown),
        "simultaneous_start_stop": _positive_max(startup + shutdown - 1),
        "fixed_on": _maximum_abs(1 - u[:, fixed]),
        "initial_dwell": _positive_max(initial_excess),
        "minimum_up": _positive_max(up_excess),
        "minimum_down": _positive_max(down_excess),
    }
    tolerances = {
        key: (
            power_tolerance
            if key.endswith(("_mw", "_mvar", "_mva"))
            else (
                voltage_tolerance
                if key.endswith("_pu")
                else angle_tolerance if key.endswith("_rad") else binary_tolerance
            )
        )
        for key in maxima
    }
    failed = [key for key, maximum in maxima.items() if maximum > tolerances[key]]
    c0, c1, c2 = [
        np.asarray([g[key] for g in generators], float)
        for key in ("cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2")
    ]
    switching = float(
        np.sum(
            startup * np.asarray(case["startup_cost"])
            + shutdown * np.asarray(case["shutdown_cost"])
        )
    )
    on_cost = _array(case.get("on_cost", [0.0] * ng), (ng,), "on_cost")
    if np.any(on_cost < 0):
        raise ValueError("Hourly online costs must be nonnegative")
    online_cost = float(np.sum(u * on_cost))
    variable_cost = float(np.sum(c2 * pg**2 + c1 * pg))
    return {
        "checker": "independent_raw_pi_branch_ac_and_uc_v1",
        "feasible": not failed,
        "failed_constraints": failed,
        "max_violations": maxima,
        "tolerances": tolerances,
        "worst_active_balance": _worst(mismatch.real),
        "worst_reactive_balance": _worst(mismatch.imag),
        "worst_ramp_up": _worst(np.maximum(ramp_up, 0)),
        "worst_ramp_down": _worst(np.maximum(ramp_down, 0)),
        "generation_variable_cost": variable_cost,
        "switching_cost": switching,
        "hourly_online_cost": online_cost,
        "online_cost_units": "cost per modeled period; current cases use one-hour periods",
        "total_cost_author_constant": variable_cost
        + switching
        + online_cost
        + nt * float(c0.sum()),
        "total_cost_commitment_constant": variable_cost
        + switching
        + online_cost
        + float(np.sum(c0 * u)),
        "periods": nt,
        "active_branches": int(physics["active"].sum()),
        "thermal_limited_branches": int(limited.sum()),
        "initial_pg_ramp_enforced": False,
        "terminal_dwell": "truncated",
        "checks_precomputed_admittances": False,
        "reference_bus_indices": reference_indices,
        "reference_angle_target_rad": 0.0,
        "silently_rotates_angles": False,
    }


def self_test():
    """Numerical oracle checks against supplied solved-state/flow labels and UC edge cases."""
    import pandas as pd

    root = Path(__file__).resolve().parents[1]
    tables = {
        kind: pd.read_parquet(
            root / "data" / "pf_case14_partition0" / f"{kind}.parquet"
        )
        for kind in ("bus", "gen", "branch")
    }
    flow_errors, balance_errors = [], []
    for scenario in sorted(tables["bus"].scenario.unique())[:5]:
        b, g, e = [
            tables[k].loc[tables[k].scenario == scenario].reset_index(drop=True)
            for k in ("bus", "gen", "branch")
        ]
        nb, ng = len(b), len(g)
        case = {
            "T": 1,
            "baseMVA": 100.0,
            "bus": b.to_dict("records"),
            "gen": g.to_dict("records"),
            "branch": e.to_dict("records"),
            "pd": [b.Pd.tolist()],
            "qd": [b.Qd.tolist()],
            "initial_on": [1] * ng,
            "min_up": [1] * ng,
            "min_down": [1] * ng,
            "startup_cost": [0] * ng,
            "shutdown_cost": [0] * ng,
            "ramp_ratio": 1.0,
        }
        solution = {
            "pg": [g.p_mw.tolist()],
            "qg": [g.q_mvar.tolist()],
            "vm": [b.Vm.tolist()],
            "va": [np.deg2rad(b.Va).tolist()],
            "uc": [[1.0] * ng],
        }
        quantities = ac_quantities(case, solution["vm"], solution["va"])
        active = quantities["active"]
        flow_error = max(
            _maximum_abs(
                quantities["from_mva"][0, active]
                - (e.pf.to_numpy() + 1j * e.qf.to_numpy())[active]
            ),
            _maximum_abs(
                quantities["to_mva"][0, active]
                - (e.pt.to_numpy() + 1j * e.qt.to_numpy())[active]
            ),
        )
        audit = check_solution(case, solution)
        balance_error = max(
            audit["max_violations"]["active_balance_mw"],
            audit["max_violations"]["reactive_balance_mvar"],
        )
        assert flow_error < 1e-6, (scenario, flow_error)
        assert balance_error < 1e-3, (scenario, balance_error)
        corrupted = copy.deepcopy(solution)
        corrupted["pg"][0][0] += 1
        corrupted_audit = check_solution(case, corrupted)
        assert abs(corrupted_audit["max_violations"]["active_balance_mw"] - 1) < 1e-3
        shifted = copy.deepcopy(solution)
        shifted["va"] = (np.asarray(shifted["va"]) + 0.1).tolist()
        shifted_audit = check_solution(case, shifted)
        assert abs(shifted_audit["max_violations"]["reference_angle_rad"] - 0.1) < 1e-12
        assert (
            abs(
                shifted_audit["max_violations"]["active_balance_mw"]
                - audit["max_violations"]["active_balance_mw"]
            )
            < 1e-10
        )
        assert "reference_angle_rad" in shifted_audit["failed_constraints"]
        flow_errors.append(flow_error)
        balance_errors.append(balance_error)

    # A phase-shifting transformer has no series current when Vf = a*Vt.
    transformer_case = {
        "baseMVA": 100.0,
        "bus": [{"bus": 0, "GS": 0.0, "BS": 0.0}, {"bus": 1, "GS": 0.0, "BS": 0.0}],
        "branch": [
            {
                "from_bus": 0,
                "to_bus": 1,
                "r": 0.01,
                "x": 0.1,
                "b": 0.0,
                "tap": 1.1,
                "shift": 15.0,
                "br_status": 1,
            }
        ],
    }
    transformer = ac_quantities(
        transformer_case, [[1.1, 1.0]], [[np.deg2rad(15.0), 0.0]]
    )
    assert _maximum_abs(transformer["from_mva"]) < 1e-10
    assert _maximum_abs(transformer["to_mva"]) < 1e-10
    transformer_case["branch"][0].update(br_status=0, r=0.0, x=0.0)
    transformer_case["bus"][1].update(GS=0.2, BS=0.3)
    shunt_only = ac_quantities(transformer_case, [[1.0, 1.0]], [[0.0, 0.0]])
    assert _maximum_abs(shunt_only["from_mva"]) == 0.0
    assert abs(shunt_only["outgoing_mva"][0, 1] - (20 - 30j)) < 1e-12

    # One-bus cases isolate scheduling semantics from network solution error.
    gen = {
        "bus": 0,
        "min_p_mw": 0.0,
        "max_p_mw": 10.0,
        "min_q_mvar": -10.0,
        "max_q_mvar": 10.0,
        "cp0_eur": 0.0,
        "cp1_eur_per_mw": 1.0,
        "cp2_eur_per_mw2": 0.0,
    }
    case = {
        "T": 3,
        "baseMVA": 100.0,
        "bus": [{"bus": 0, "GS": 0.0, "BS": 0.0, "min_vm_pu": 0.9, "max_vm_pu": 1.1}],
        "gen": [gen],
        "branch": [],
        "pd": [[0.0], [0.0], [0.0]],
        "qd": [[0.0], [0.0], [0.0]],
        "initial_on": [0],
        "min_up": [2],
        "min_down": [2],
        "startup_cost": [10],
        "shutdown_cost": [2],
        "ramp_ratio": 0.5,
    }
    solution = {
        "pg": [[0.0], [0.0], [0.0]],
        "qg": [[0.0], [0.0], [0.0]],
        "vm": [[1.0], [1.0], [1.0]],
        "va": [[0.0], [0.0], [0.0]],
        "uc": [[0.0], [0.0], [1.0]],
    }
    assert check_solution(case, solution)[
        "feasible"
    ]  # Last-period startup is allowed by truncation.
    solution["uc"] = [[1.0], [0.0], [0.0]]
    assert "minimum_up" in check_solution(case, solution)["failed_constraints"]
    solution["uc"] = [[1.0], [1.0], [0.5]]
    assert "integrality" in check_solution(case, solution)["failed_constraints"]
    case["initial_on"] = [1]
    case["initial_forced_on"] = [2]
    solution["uc"] = [[1.0], [0.0], [1.0]]
    failed = check_solution(case, solution)["failed_constraints"]
    assert "initial_dwell" in failed and "minimum_down" in failed
    case["initial_on"] = [0]
    case["initial_forced_on"] = [0]
    case["initial_forced_off"] = [2]
    solution["uc"] = [[0.0], [1.0], [1.0]]
    assert "initial_dwell" in check_solution(case, solution)["failed_constraints"]
    case["initial_forced_off"] = [0]
    case["fixed_on"] = [0]
    solution["uc"] = [[0.0], [1.0], [1.0]]
    assert "fixed_on" in check_solution(case, solution)["failed_constraints"]
    case["fixed_on"] = []
    solution["pg"] = [[0.0], [6.0], [0.0]]
    case["pd"] = copy.deepcopy(solution["pg"])
    solution["uc"] = [[0.0], [1.0], [0.0]]
    ramps = check_solution(case, solution)["max_violations"]
    assert ramps["ramp_up_mw"] == 1 and ramps["ramp_down_mw"] == 1
    solution["pg"][0][0] = float("nan")
    try:
        check_solution(case, solution)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite solution was accepted")
    return {
        "status": "passed",
        "official_scenarios": len(flow_errors),
        "max_branch_flow_error_against_labels_mva": max(flow_errors),
        "max_ac_balance_error_on_labels_mva": max(balance_errors),
        "intentional_1mw_error_detected": True,
        "intentional_global_angle_shift_detected": True,
        "phase_shifter_shunt_and_offline_branch_checks": "passed",
        "uc_tests": [
            "terminal_truncation",
            "min_up",
            "min_down",
            "initial_dwell",
            "fixed_on",
            "integrality",
            "startup_ramp",
            "shutdown_ramp",
            "nonfinite_rejection",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instance", nargs="?")
    parser.add_argument("solution", nargs="?")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--power-tolerance", type=float, default=1e-3)
    args = parser.parse_args()
    if args.self_test:
        result = self_test()
    else:
        if not args.instance or not args.solution:
            parser.error("Provide instance.json and solution.json, or --self-test")
        case = json.loads(Path(args.instance).read_text())
        solution = json.loads(Path(args.solution).read_text())
        result = check_solution(case, solution, power_tolerance=args.power_tolerance)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
