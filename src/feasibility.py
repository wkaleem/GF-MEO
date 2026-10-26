"""UC–ACOPF feasibility checks from generation and bus voltages."""

import cmath
import math


def finite_data(value):
    if isinstance(value, dict):
        return all(finite_data(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_data(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def check(
    case, solution, power_tolerance=1e-3, voltage_tolerance=1e-5, angle_tolerance=1e-6
):
    tolerances = (power_tolerance, voltage_tolerance, angle_tolerance)
    if any(not math.isfinite(v) or v <= 0 for v in tolerances):
        raise ValueError("Positive finite tolerances required")
    if not finite_data(case):
        raise ValueError("Nonfinite instance data")
    T = case["T"]
    G = len(case["gen"])
    N = len(case["bus"])
    base = case["baseMVA"]
    if T < 1 or not math.isfinite(base) or base <= 0:
        raise ValueError("Invalid horizon or power base")
    for k, n in [
        ("u", G),
        ("su", G),
        ("sd", G),
        ("pg", G),
        ("qg", G),
        ("vm", N),
        ("va", N),
    ]:
        a = solution[k]
        if len(a) != T or any(
            len(row) != n or any(not math.isfinite(v) for v in row) for row in a
        ):
            return dict(
                accepted=False,
                reason="invalid solution dimensions or nonfinite values",
                field=k,
            )
    binary = all(
        v in (0.0, 1.0) for k in ("u", "su", "sd") for row in solution[k] for v in row
    )
    u, su, sd, pg, qg, vm, va = (
        solution[k] for k in ("u", "su", "sd", "pg", "qg", "vm", "va")
    )
    maximum = {}
    ratios = {}

    def violation(name, value, tol):
        if not math.isfinite(value):
            raise ValueError("Nonfinite " + name)
        maximum[name] = max(maximum.get(name, 0.0), value, 0.0)
        ratios[name] = maximum[name] / tol

    ids = {b["bus"]: i for i, b in enumerate(case["bus"])}
    if len(ids) != N:
        raise ValueError("Duplicate bus IDs")
    principal = True
    objective = 0.0
    for t in range(T):
        outgoing = [0j] * N
        voltages = [vm[t][i] * cmath.exp(1j * va[t][i]) for i in range(N)]
        for line in case["branch"]:
            if line["br_status"] not in (0, 1):
                raise ValueError("Nonbinary line status")
            if not line["br_status"]:
                continue
            i, j = ids[line["from_bus"]], ids[line["to_bus"]]
            z = complex(line["r"], line["x"])
            tap = line["tap"] or 1.0
            if z == 0 or tap <= 0:
                raise ValueError("Invalid in-service line impedance or tap")
            ratio = tap * cmath.exp(1j * math.radians(line["shift"]))
            ys = 1 / z
            charging = 0.5j * line["b"]
            vf, vt = voltages[i], voltages[j]
            sf = (
                base
                * vf
                * (
                    ((ys + charging) / abs(ratio) ** 2) * vf
                    - ys / ratio.conjugate() * vt
                ).conjugate()
            )
            st = base * vt * ((ys + charging) * vt - ys / ratio * vf).conjugate()
            outgoing[i] += sf
            outgoing[j] += st
            if line["rate_a"] <= 0:
                raise ValueError("Active-line ratings must be positive")
            violation(
                "line_mva", max(abs(sf), abs(st)) - line["rate_a"], power_tolerance
            )
            delta = va[t][i] - va[t][j]
            principal = principal and abs(delta) < math.pi / 2
            violation(
                "angle_relation_rad",
                abs(delta - math.atan(math.tan(delta))),
                angle_tolerance,
            )
        pbus = [0.0] * N
        qbus = [0.0] * N
        for g, gen in enumerate(case["gen"]):
            i = ids[gen["bus"]]
            pbus[i] += pg[t][g]
            qbus[i] += qg[t][g]
            violation(
                "p_bounds_mw",
                max(
                    -pg[t][g],
                    gen["min_p_mw"] * u[t][g] - pg[t][g],
                    pg[t][g] - gen["max_p_mw"] * u[t][g],
                ),
                power_tolerance,
            )
            violation(
                "q_bounds_mvar",
                max(
                    -qg[t][g],
                    gen["min_q_mvar"] * u[t][g] - qg[t][g],
                    qg[t][g] - gen["max_q_mvar"] * u[t][g],
                ),
                power_tolerance,
            )
            if t:
                violation(
                    "transition", abs(u[t - 1][g] - u[t][g] + su[t][g] - sd[t][g]), 1e-5
                )
            if t < case["initial_forced_on"][g]:
                violation("initial_obligations", 1 - u[t][g], 1e-5)
            if t < case["initial_forced_off"][g]:
                violation("initial_obligations", u[t][g], 1e-5)
            U, D = case["min_up"][g], case["min_down"][g]
            if U < 1 or D < 1 or int(U) != U or int(D) != D:
                raise ValueError("Positive integer minimum durations required")
            if t >= U - 1:
                violation(
                    "minimum_up",
                    sum(su[k][g] for k in range(t - U + 1, t + 1)) - u[t][g],
                    1e-5,
                )
            if t >= D - 1:
                violation(
                    "minimum_down",
                    sum(sd[k][g] for k in range(t - D + 1, t + 1)) - 1 + u[t][g],
                    1e-5,
                )
            p0 = case["initial_pg_mw"][g] if t == 0 else pg[t - 1][g]
            u0 = case["initial_on"][g] if t == 0 else u[t - 1][g]
            violation(
                "ramp_up_mw",
                pg[t][g]
                - p0
                - case["ramp_up_mw"][g] * u0
                - case["startup_ramp_mw"][g] * su[t][g],
                power_tolerance,
            )
            violation(
                "ramp_down_mw",
                p0
                - pg[t][g]
                - case["ramp_down_mw"][g] * u[t][g]
                - case["shutdown_ramp_mw"][g] * sd[t][g],
                power_tolerance,
            )
            objective += (
                gen["cp0_eur"]
                + gen["cp1_eur_per_mw"] * pg[t][g]
                + gen["cp2_eur_per_mw2"] * pg[t][g] ** 2
            )
            objective += (
                case["on_cost"][g] * u[t][g]
                + case["startup_cost"][g] * su[t][g]
                + case["shutdown_cost"][g] * sd[t][g]
            )
        for i, bus in enumerate(case["bus"]):
            outgoing[i] += (
                case.get("shunt_baseMVA", 100.0)
                * complex(bus["GS"], -bus["BS"])
                * vm[t][i] ** 2
            )
            violation(
                "p_balance_mw",
                abs(pbus[i] - case["pd"][t][i] - outgoing[i].real),
                power_tolerance,
            )
            violation(
                "q_balance_mvar",
                abs(qbus[i] - case["qd"][t][i] - outgoing[i].imag),
                power_tolerance,
            )
            violation(
                "voltage_pu",
                max(bus["min_vm_pu"] - vm[t][i], vm[t][i] - bus["max_vm_pu"]),
                voltage_tolerance,
            )
            if bus["REF"]:
                violation("reference_rad", abs(va[t][i]), angle_tolerance)
    if not math.isfinite(objective):
        raise ValueError("Nonfinite original objective")
    return dict(
        accepted=binary and principal and max(ratios.values(), default=0.0) <= 1.0,
        exact_binary=binary,
        principal_angle_domain=principal,
        objective=objective,
        max_violations=maximum,
        max_normalized_violation=max(ratios.values(), default=0.0),
        tolerances=dict(
            power=power_tolerance, voltage=voltage_tolerance, angle=angle_tolerance
        ),
        specification="UC–ACOPF constraints with initial dispatch ramps",
    )
