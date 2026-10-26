"""Augmented Lagrangian decomposition for UC–ACOPF."""

from __future__ import annotations
import time

PROCESS_STARTED = time.monotonic()
import argparse
import hashlib
import itertools
import json
from pathlib import Path
from contextlib import ExitStack
import sys
import numpy as np
from uc_subproblem import arrays, check_horizon, economic_cost, solve_horizon
from lagrangian import period_objective, horizon_objective
from networks import network_spec
from ac_solver import DirectPeriodBackend
from ac_solver_session import solve_direct_period, WorkerError
from feasibility import check as check_full_solution


def load_opf_bundle(**kwargs):
    from genco_model import load_opf_bundle as load

    return load(**kwargs)


def checkpoint_files(*args, **kwargs):
    from genco_model import checkpoint_files as files

    return files(*args, **kwargs)


def CorrectedPeriodOracle(*args, **kwargs):
    from ac_subproblem import CorrectedPeriodOracle as oracle

    return oracle(*args, **kwargs)


def solve_corrected_period(*args, **kwargs):
    from ac_subproblem import solve_corrected_period as solve

    return solve(*args, **kwargs)


def atomic_json(path, record):
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        torch = sys.modules.get("torch")
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy().tolist()
        raise TypeError(type(value).__name__)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(record, default=serial, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


def interface(solution):
    return np.concatenate((solution["u"], solution["p"]), axis=1)


def economic_penalty_scale(case):
    """Original-cost gradient scale for dimensionless horizon variables.

    Dispatch is P/baseMVA, hence dC/dp = baseMVA*(c1+2*c2*P).
    Endpoint maxima apply even when a quadratic coefficient is negative.
    Additive generation constants do not contribute to this gradient.
    """
    _, _, base, data = arrays(case)
    c1 = np.asarray([g["cp1_eur_per_mw"] for g in case["gen"]])
    c2 = np.asarray([g["cp2_eur_per_mw2"] for g in case["gen"]])
    maximum = base * data["pmax"]
    scale = float(
        np.max(
            np.abs(
                np.concatenate(
                    (
                        base * c1,
                        base * (c1 + 2 * c2 * maximum),
                        data["on_cost"],
                        data["startup_cost"],
                        data["shutdown_cost"],
                    )
                )
            )
        )
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive finite economic gradient scale required")
    return scale


def resolve_penalty(config, case=None):
    value = config["rho"]
    policy = config.get("penalty_update")
    if value == "economic_gradient_scale":
        rho = economic_penalty_scale(case)
        if not isinstance(policy, dict) or set(policy) != {
            "factor",
            "contraction",
            "maximum_multiple",
        }:
            raise ValueError(
                "Economic scaling requires factor, contraction and maximum_multiple"
            )
        multiple = policy["maximum_multiple"]
        if (
            isinstance(multiple, bool)
            or not isinstance(multiple, (int, float))
            or not np.isfinite(multiple)
            or multiple < 1
        ):
            raise ValueError("Finite maximum_multiple >= 1 required")
        policy = dict(
            factor=policy["factor"],
            contraction=policy["contraction"],
            maximum=rho * multiple,
        )
    else:
        rho = float(value)
    if not np.isfinite(rho) or rho <= 0:
        raise ValueError("Positive finite rho required")
    validate_penalty_policy(policy, rho)
    return rho, policy


def validate_penalty_policy(policy, rho):
    """Validate a positive, bounded penalty update."""
    if policy is None:
        return
    if not isinstance(policy, dict) or set(policy) != {
        "factor",
        "contraction",
        "maximum",
    }:
        raise ValueError("penalty_update requires factor, contraction and maximum")
    factor, contraction, maximum = (
        policy[k] for k in ("factor", "contraction", "maximum")
    )
    if any(
        isinstance(v, bool) or not isinstance(v, (int, float))
        for v in (factor, contraction, maximum)
    ):
        raise ValueError("Numeric penalty_update parameters required")
    if not np.isfinite([factor, contraction, maximum]).all():
        raise ValueError("Finite penalty_update parameters required")
    if factor <= 1 or not 0 < contraction < 1 or maximum < rho:
        raise ValueError(
            "penalty_update requires factor>1, 0<contraction<1, maximum>=rho"
        )


def next_penalty(rho, previous_residual, residual, tolerance, policy):
    """Update the penalty for the next iteration."""
    if policy is None:
        return rho, "fixed"
    if (
        not np.isfinite([previous_residual, residual]).all()
        or min(previous_residual, residual) < 0
    ):
        raise ValueError("Finite nonnegative consensus residuals required")
    if residual <= max(tolerance, policy["contraction"] * previous_residual):
        return rho, "sufficient_consensus_progress"
    maximum = policy["maximum"]
    if rho >= maximum:
        return rho, "penalty_cap"
    # Avoid overflow in factor*rho before clamping.
    updated = maximum if rho >= maximum / policy["factor"] else policy["factor"] * rho
    return updated, "insufficient_consensus_progress"


def patterns(case):
    """Enumerate units not fixed online or offline throughout this instance."""
    T, G, _, data = arrays(case)
    fixed = np.flatnonzero(data["initial_forced_on"] == T)
    fixed_off = np.flatnonzero(data["initial_forced_off"] == T)
    if np.intersect1d(fixed, fixed_off).size:
        raise ValueError("Conflicting fixed online/offline obligations")
    free = np.array(
        [g for g in range(G) if g not in fixed and g not in fixed_off], dtype=int
    )
    if len(free) > 12:
        raise ValueError("At most 12 switchable generators are supported")
    result = []
    for bits in itertools.product((0.0, 1.0), repeat=len(free)):
        u = np.ones(G)
        u[fixed_off] = 0.0
        u[free] = bits
        result.append(u)
    return result


def check_reconstruction(stored, fresh):
    """Check prediction identity up to float64 reduction roundoff, not feasibility.

    Physical acceptance is checked separately on the fresh prediction. These
    absolute comparison thresholds are much smaller than physical tolerances.
    Binary indicators and the subsequent fixed-dispatch horizon remain exact.
    """
    tolerances = {"pg": 1e-8, "qg": 1e-8, "vm": 1e-10, "va": 1e-10}
    errors = {}
    for key in ("uc", "pg", "qg", "vm", "va"):
        old = np.asarray(stored[key], float)
        new = np.asarray(fresh[key], float)
        if (
            old.shape != new.shape
            or not np.isfinite(old).all()
            or not np.isfinite(new).all()
        ):
            raise ValueError(
                "Neural reconstruction has inconsistent shape or nonfinite state"
            )
        error = float(np.max(np.abs(new - old), initial=0.0))
        if (
            key == "uc"
            and (not np.array_equal(new, old) or not np.isin(new, [0.0, 1.0]).all())
        ) or (key != "uc" and error > tolerances[key]):
            raise ValueError("Neural reconstruction differs from the stored candidate")
        errors[key] = error
    return dict(
        max_absolute_errors=errors,
        absolute_tolerances=tolerances,
        units={"pg": "MW", "qg": "MVAr", "vm": "p.u.", "va": "rad"},
        exact_binary=True,
        accepted_state="fresh independently checked reconstruction",
    )


def complete(case, oracles, periods, time_limit):
    """Reconstruct period states, then fix their actual u,p for UC completion."""
    started = time.monotonic()
    base = case["baseMVA"]
    if len(oracles) != case["T"] or len(periods) != case["T"]:
        raise ValueError("Completion requires one oracle and point for every period")
    states = []
    checks = []
    records = []
    reconstructions = []
    physical_keys = (
        "T",
        "baseMVA",
        "bus",
        "gen",
        "branch",
        "pd",
        "qd",
        "shunt_baseMVA",
    )
    for t, (oracle, point) in enumerate(zip(oracles, periods)):
        if time.monotonic() - started >= time_limit:
            return None
        if oracle.t != t or any(
            oracle.case.get(k) != case.get(k) for k in physical_keys
        ):
            raise ValueError(
                "Period oracle does not match the completion period and original network"
            )
        oracle.cache = None
        oracle.deadline = started + time_limit
        direct = point.get("coordinate_kind") == "physical"
        try:
            entry = oracle.evaluate(
                point["physical_coordinates"] if direct else point["z"]
            )
            checked = oracle.check(entry)
        except (TimeoutError, FloatingPointError):
            return None
        if time.monotonic() - started >= time_limit:
            return None
        if not checked["accepted_period"]:
            return None
        reconstruction = check_reconstruction(point["state"], entry["state"])
        if direct:
            reconstruction["accepted_state"] = (
                "stored physical coordinates, independently rechecked without AC redispatch"
            )
        reconstructions.append(reconstruction)
        states.append(entry["state"])
        checks.append(checked)
        keys = (
            ("state", "interface", "coordinate_kind", "physical_coordinates")
            if direct
            else ("z", "x", "state", "interface", "raw_prediction", "correction")
        )
        records.append({k: entry[k] for k in keys if k in entry})
    u = np.stack([s["uc"] for s in states])
    p = np.stack([s["pg"] / base for s in states])
    remaining = time_limit - (time.monotonic() - started)
    if remaining <= 0:
        return None
    # Generation cost is constant at fixed p: the minimizer is the UC completion.
    result = solve_horizon(case, fixed_u=u, fixed_p=p, time_limit=remaining)
    if result["solution"] is None or time.monotonic() - started >= time_limit:
        return None
    solution = result["solution"]
    checked_horizon = check_horizon(case, solution)
    if not checked_horizon["accepted"]:
        return None
    if not np.array_equal(solution["u"], u) or not np.array_equal(solution["p"], p):
        raise ValueError(
            "Completion changed prescribed online indicators or generation"
        )
    witness = {k: solution[k].tolist() for k in ("u", "su", "sd")}
    witness.update(
        {k: [state[k].tolist() for state in states] for k in ("pg", "qg", "vm", "va")}
    )
    full_check = check_full_solution(case, witness)
    if not full_check["accepted"] or time.monotonic() - started >= time_limit:
        return None
    return dict(
        objective=economic_cost(case, solution),
        horizon=solution,
        periods=records,
        horizon_check=checked_horizon,
        period_checks=checks,
        reconstruction_checks=reconstructions,
        full_check=full_check,
        acceptance="UC–ACOPF constraints at the stated tolerances; exact binary decisions",
        global_lower_bound=None,
        certified_gap=None,
    )


def run(config, *, started=None):
    with ExitStack() as resources:
        return _run(config, started=started, resources=resources)


def _run(config, *, started, resources):
    if config.get("model_device", "cpu") != "cpu":
        raise ValueError("Only CPU model execution is supported")
    started = time.monotonic() if started is None else started
    deadline = started + float(config["seconds"])
    variant = config.get("period_model", "corrected")
    if variant not in ("corrected", "direct"):
        raise ValueError("Unsupported period model")
    direct = variant == "direct"
    if direct and any(
        k in config
        for k in (
            "radius",
            "maxiter",
            "initialization_maxiter",
            "input_updates",
            "optimize_linear_costs",
            "model_size",
            "training_seed",
        )
    ):
        raise ValueError(
            "Direct period model does not accept neural input or gradient settings"
        )
    radius = None if direct else float(config["radius"])
    if config["rho"] != "economic_gradient_scale":
        resolve_penalty(config)
    if not np.isfinite(config["seconds"]) or config["seconds"] <= 0:
        raise ValueError("Positive finite total budget required")
    for name in ("period_seconds", "horizon_seconds", "completion_seconds"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name)
    iteration_limit = config["iterations"]
    if iteration_limit is not None and (
        type(iteration_limit) is not int or iteration_limit < 1
    ):
        raise ValueError("iterations must be a positive integer or null")
    for name in (
        ("initialization_starts",) if direct else ("initialization_starts", "maxiter")
    ):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(name)
    for name in ("consensus_tolerance", "step_tolerance"):
        if not np.isfinite(config[name]) or config[name] <= 0:
            raise ValueError(name)
    initialization_seconds = config.get(
        "initialization_seconds", config["period_seconds"]
    )
    initialization_maxiter = (
        None if direct else config.get("initialization_maxiter", config["maxiter"])
    )
    if not np.isfinite(initialization_seconds) or initialization_seconds <= 0:
        raise ValueError("initialization_seconds")
    if not direct and (
        not isinstance(initialization_maxiter, int) or initialization_maxiter < 1
    ):
        raise ValueError("initialization_maxiter")
    input_updates = config.get("input_updates", "optimize")
    if input_updates not in ("optimize", "fixed_after_initialization"):
        raise ValueError("input_updates must be optimize or fixed_after_initialization")
    fixed_inputs = input_updates == "fixed_after_initialization"
    optimize_costs = config.get("optimize_linear_costs", True)
    if not isinstance(optimize_costs, bool):
        raise ValueError("optimize_linear_costs must be boolean")
    compiled_cache = {}
    backend = None

    def period_factory(t, u):
        if direct:
            return backend.oracle(t, u)
        return CorrectedPeriodOracle(
            case,
            t,
            u,
            radius,
            "literal_period",
            optimize_linear_costs=optimize_costs,
            model_bundle=bundle,
            compiled_cache=compiled_cache,
            deadline=deadline,
        )

    period_solver = solve_direct_period if direct else solve_corrected_period
    path = Path(config["case"])
    case = json.loads(path.read_text())
    T, G, _, data = arrays(case)
    w = np.ones(2 * G)
    lam = np.zeros((T, 2 * G))
    rho, penalty_policy = resolve_penalty(config, case)
    rng = np.random.default_rng(config["seed"])
    if not direct:
        import torch

        torch.set_num_threads(1)
    report = dict(
        status="initializing",
        implementation=(
            "direct_ac_decomposition" if direct else "corrected_decomposition"
        ),
        benchmark_validated=False,
        configuration=config,
        case_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        best=None,
        first_accepted_at_s=None,
        accepted_history=[],
        history=[],
        calls=[],
        failures=[],
        input_domains=[],
        initial_inputs=[],
        scope="GFMO UC–ACOPF solve",
        global_lower_bound=None,
        certified_gap=None,
        termination_reason=None,
        initial_consensus=None,
        penalty_mode="fixed" if penalty_policy is None else "bounded_residual_progress",
    )
    report["resolved_penalty"] = dict(
        initial=rho, update=penalty_policy, rule=config["rho"]
    )
    report["period_model"] = dict(
        variant=variant,
        optimize_linear_costs=optimize_costs,
        input_updates=input_updates,
        relation="trained-model outputs followed by checked local physical correction; replaces raw generation/voltage/reactive links",
        deadline_policy="Checks between atomic library calls; late evaluations cannot create incumbents.",
    )
    root = Path(__file__).resolve().parents[1]
    spec = network_spec(case)
    if direct:
        backend = DirectPeriodBackend(
            case,
            path,
            config["export_folder"],
            config.get(
                "worker_directory",
                str(Path(config["output"]).with_suffix("")) + "_workers",
            ),
            deadline=deadline,
        )
        resources.callback(backend.close)
        report["worker_sessions"] = backend.sessions
        report["direct_export"] = backend.export_record()
        report["period_model"] = dict(
            variant="direct",
            coordinate_kind="physical",
            checkpoint_loaded=False,
            relation="physical AC period optimization; no trained model or input optimization",
            initialization="horizon dispatch clipped to period bounds, midpoint Qg/Vm, zero Va; bounded random physical starts on later attempts",
            restart_policy="restart a closed worker before the next attempt; startup and restart consume the global budget",
        )
        report.pop("initial_inputs")
        report.pop("input_domains")
        report["period_domains"] = []
        report["initial_physical_points"] = []
        artifacts = [
            Path(__file__).with_name(name)
            for name in (
                "decomposition.py",
                "uc_subproblem.py",
                "lagrangian.py",
                "ac_solver.py",
                "ac_solver_session.py",
                "networks.py",
                "feasibility.py",
                "ac_constraints.py",
            )
        ]
        artifacts.append(root / spec.nominal_path)
    else:
        model_size = config.get("model_size", "small")
        training_seed = config.get("training_seed", 0)
        model_files = checkpoint_files(spec.name, model_size, training_seed)
        artifacts = [Path(record["local_path"]) for record in model_files.values()]
        report["checkpoint_selection"] = dict(
            network=spec.name,
            model_size=model_size,
            training_seed=training_seed,
            files=model_files,
        )
        artifacts += [
            Path(__file__).with_name(name)
            for name in (
                "decomposition.py",
                "uc_subproblem.py",
                "lagrangian.py",
                "model_evaluation.py",
                "genco_model.py",
                "model_inputs.py",
                "ac_constraints.py",
                "ac_derivatives.py",
                "input_domain.py",
                "genco_models.json",
                "networks.py",
                "feasibility.py",
                "reactive_generation.py",
            )
        ]
        artifacts += [
            Path(__file__).with_name(name)
            for name in (
                "ac_subproblem.py",
                "ac_correction.py",
                "ac_solver.py",
                "ac_solver_session.py",
            )
        ]
    report["artifacts"] = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in artifacts
    }
    if direct:
        report["artifacts"].update(backend.artifacts())

    def remaining(cap):
        return min(cap, max(0.0, deadline - time.monotonic()))

    def save():
        report["elapsed_s"] = time.monotonic() - started
        atomic_json(config["output"], report)

    def accept(selected, index):
        budget = remaining(config["completion_seconds"])
        if budget <= 0:
            return False
        candidate = complete(
            case, [pair[0] for pair in selected], [pair[1] for pair in selected], budget
        )
        accepted_time = time.monotonic()
        if candidate is not None and accepted_time < deadline:
            candidate["accepted_at_s"] = accepted_time - started
            candidate["iteration"] = index
            if report["first_accepted_at_s"] is None:
                report["first_accepted_at_s"] = candidate["accepted_at_s"]
            if (
                report["best"] is None
                or candidate["objective"] < report["best"]["objective"]
            ):
                report["best"] = candidate
            report["accepted_history"].append(
                dict(
                    iteration=index,
                    elapsed_s=candidate["accepted_at_s"],
                    objective=candidate["objective"],
                    best_objective=report["best"]["objective"],
                )
            )
            save()
            return True
        return False

    save()
    initial_budget = remaining(config["horizon_seconds"])
    initial = (
        solve_horizon(case, time_limit=initial_budget)
        if initial_budget > 0
        else {"solution": None}
    )
    horizon = initial["solution"]
    if horizon is None:
        report["status"] = "no_horizon_initial_point"
        report["termination_reason"] = (
            "time_limit" if remaining(1) <= 0 else "horizon_initialization_failed"
        )
        save()
        return report
    bundle = (
        None
        if direct
        else load_opf_bundle(
            network=spec.name,
            model_size=model_size,
            training_seed=training_seed,
            device="cpu",
        )
    )
    report["model_execution"] = (
        dict(device="cpu", checkpoint_loaded=False)
        if direct
        else (
            dict(
                device=str(next(bundle[1].parameters()).device),
                dtype=str(next(bundle[1].parameters()).dtype),
                optimizer_and_physical_check_device="cpu",
            )
            if bundle is not None
            else None
        )
    )
    all_patterns = patterns(case)
    report["pattern_domain"] = dict(
        enumerated=all_patterns,
        initial_obligations="Full-horizon online/offline obligations fix the corresponding period indicators.",
        released_adapter_restrictions=f"Reference generators online; one to {spec.maximum_outages} sampled generator/line outages beyond native offline equipment; connected {spec.display} network.",
        scope="Commitment patterns satisfy fixed-unit and equipment-availability restrictions.",
    )
    oracle_table = {}
    available = {}
    selected = []
    initial_inputs = {}

    def record_initial_input(key, coordinates, point):
        if direct:
            if key not in initial_inputs:
                initial_inputs[key] = coordinates
                report["initial_physical_points"].append(
                    dict(period=key[0], pattern=key[1], coordinates=coordinates)
                )
            return
        if key not in initial_inputs:
            saved = np.array(coordinates, dtype=float, copy=True)
            saved.setflags(write=False)
            initial_inputs[key] = saved
            report["initial_inputs"].append(
                dict(period=key[0], pattern=key[1], z=saved)
            )
        if fixed_inputs and not np.array_equal(point["z"], initial_inputs[key]):
            raise ValueError(
                "Fixed model inputs changed after their first feasible initialization"
            )

    # Initialize each period independently; these points need not agree with H.
    for t in range(T):
        chosen = None
        ordered = sorted(
            all_patterns,
            key=lambda u: np.count_nonzero(u - (data["initial_on"] if t == 0 else 1)),
        )
        for u in ordered:
            if remaining(1) <= 0:
                break
            key = (t, tuple(u))
            try:
                oracle = period_factory(t, u)
            except TimeoutError as error:
                report["failures"].append(
                    dict(period=t, pattern=u, phase="oracle setup", reason=str(error))
                )
                break
            except (ValueError, NotImplementedError) as error:
                report["failures"].append(
                    dict(
                        period=t, pattern=u, phase="input validation", reason=str(error)
                    )
                )
                continue
            oracle_table[key] = oracle
            report["period_domains" if direct else "input_domains"].append(
                backend.domain_record(oracle) if direct else oracle.domain_record()
            )
            for attempt in range(config["initialization_starts"]):
                if direct:
                    try:
                        backend.prepare(oracle)
                    except (TimeoutError, WorkerError, OSError) as error:
                        report["failures"].append(
                            dict(
                                period=t,
                                pattern=u,
                                phase="worker setup",
                                reason=str(error),
                            )
                        )
                        break
                budget = remaining(initialization_seconds)
                if budget <= 0:
                    break
                z = (
                    backend.initial(oracle, horizon, attempt, rng)
                    if direct
                    else (
                        oracle.z0
                        if attempt == 0
                        else rng.uniform(-0.75, 0.75, len(oracle.free))
                    )
                )
                result = period_solver(
                    oracle,
                    target=np.zeros(2 * G),
                    multipliers=np.zeros(2 * G),
                    rho=rho,
                    weights=w,
                    time_limit=budget,
                    initial=z,
                    feasibility_only=True,
                    **({} if direct else {"maxiter": initialization_maxiter}),
                )
                report["calls"].append(
                    dict(
                        phase="initialization",
                        period=t,
                        pattern=u,
                        attempt=attempt,
                        status=result["solver_status"],
                        elapsed_s=result["elapsed_s"],
                        iterations=result.get("iterations"),
                        checked_points=result.get("checked_points"),
                        closest_checked_point=result.get("closest_checked_point"),
                        failures=result.get("failures", []),
                    )
                )
                save()
                if result["candidate"] is not None:
                    point = result["candidate"]["entry"]
                    record_initial_input(key, z, point)
                    available[key] = point
                    chosen = (oracle, point)
                    break
            if chosen is not None:
                break
        if chosen is None:
            report["status"] = "period_initialization_unsuccessful"
            report["termination_reason"] = (
                "time_limit" if remaining(1) <= 0 else "period_initialization_failed"
            )
            save()
            return report
        selected.append(chosen)
        save()
    previous_residual = float(
        np.max(
            np.sqrt(w)
            * np.abs(
                np.stack([point["interface"] for _, point in selected])
                - interface(horizon)
            )
        )
    )
    report["initial_consensus"] = previous_residual
    accept(selected, 0)
    report["status"] = "iterating"
    # With no iteration cap, the time budget and convergence tests apply.
    outer_indices = (
        itertools.count() if iteration_limit is None else range(iteration_limit)
    )
    for n in outer_indices:
        if remaining(1) <= 0:
            break
        old_fm = np.stack([point["interface"] for _, point in selected])
        budget = remaining(config["horizon_seconds"])
        if budget <= 0:
            break
        result = solve_horizon(
            case, target=old_fm, multipliers=lam, rho=rho, weights=w, time_limit=budget
        )
        candidates = [horizon]
        if result["solution"] is not None:
            candidates.append(result["solution"])
        horizon = min(
            candidates,
            key=lambda s: horizon_objective(
                economic_cost(case, s), interface(s), old_fm, lam, rho, w
            ),
        )
        uc = interface(horizon)
        report["calls"].append(
            dict(
                phase="horizon",
                iteration=n + 1,
                status=result["solver_status"],
                elapsed_s=result["elapsed_s"],
            )
        )
        next_selected = []
        for t in range(T):
            # Reevaluate the preceding period iterate under the current objective.
            choices = [selected[t]]
            for u in all_patterns:
                budget = remaining(config["period_seconds"])
                if budget <= 0:
                    break
                key = (t, tuple(u))
                if key not in oracle_table:
                    try:
                        oracle_table[key] = period_factory(t, u)
                        report["period_domains" if direct else "input_domains"].append(
                            backend.domain_record(oracle_table[key])
                            if direct
                            else oracle_table[key].domain_record()
                        )
                    except TimeoutError as error:
                        report["failures"].append(
                            dict(
                                period=t,
                                pattern=u,
                                phase="oracle setup",
                                reason=str(error),
                            )
                        )
                        break
                    except (ValueError, NotImplementedError) as error:
                        report["failures"].append(
                            dict(
                                period=t,
                                pattern=u,
                                phase="input validation",
                                reason=str(error),
                            )
                        )
                        continue
                oracle = oracle_table[key]
                previous = available.get(key)
                if previous is not None:
                    choices.append((oracle, previous))
                if direct:
                    try:
                        backend.prepare(oracle)
                    except (TimeoutError, WorkerError, OSError) as error:
                        report["failures"].append(
                            dict(
                                period=t,
                                pattern=u,
                                phase="worker setup",
                                reason=str(error),
                            )
                        )
                        break
                budget = remaining(config["period_seconds"])
                if budget <= 0:
                    break
                start = (
                    (
                        backend.initial(oracle, horizon, 0, rng)
                        if previous is None
                        else previous["physical_coordinates"]
                    )
                    if direct
                    else None if previous is None else previous["z"]
                )
                result = period_solver(
                    oracle,
                    target=uc[t],
                    multipliers=lam[t],
                    rho=rho,
                    weights=w,
                    time_limit=budget,
                    previous=previous,
                    initial=start,
                    feasibility_only=fixed_inputs,
                    **({} if direct else {"maxiter": config["maxiter"]}),
                )
                report["calls"].append(
                    dict(
                        phase="period",
                        iteration=n + 1,
                        period=t,
                        pattern=u,
                        status=result["solver_status"],
                        elapsed_s=result["elapsed_s"],
                        iterations=result.get("iterations"),
                        checked_points=result.get("checked_points"),
                        closest_checked_point=result.get("closest_checked_point"),
                        failures=result.get("failures", []),
                        accepted_input_steps=result.get("accepted_steps", []),
                    )
                )
                if result["candidate"] is not None:
                    point = result["candidate"]["entry"]
                    record_initial_input(
                        key,
                        (
                            start
                            if direct
                            else oracle.z0 if previous is None else previous["z"]
                        ),
                        point,
                    )
                    available[key] = point
                    choices.append((oracle, point))
            next_selected.append(
                min(
                    choices,
                    key=lambda pair: period_objective(
                        pair[1]["interface"], uc[t], lam[t], rho, w
                    ),
                )
            )
        if remaining(1) <= 0:
            break  # return the already saved incumbent
        selected = next_selected
        fm = np.stack([point["interface"] for _, point in selected])
        accepted = accept(selected, n + 1)
        residual = fm - uc
        lam = lam + rho * w * residual
        if not np.isfinite(lam).all():
            raise FloatingPointError("Nonfinite consensus multipliers")
        con = float(np.max(np.sqrt(w) * np.abs(residual)))
        step = float(np.max(np.sqrt(w) * np.abs(fm - old_fm)))
        stopping = (
            accepted
            and con <= config["consensus_tolerance"]
            and step <= config["step_tolerance"]
        )
        last_iteration = iteration_limit is not None and n + 1 >= iteration_limit
        if stopping or remaining(1) <= 0 or last_iteration:
            penalty_reason = (
                "accepted_stopping_criteria"
                if stopping
                else "time_limit" if remaining(1) <= 0 else "iteration_limit"
            )
            rho_next = rho
        else:
            rho_next, penalty_reason = next_penalty(
                rho,
                previous_residual,
                con,
                config["consensus_tolerance"],
                penalty_policy,
            )
        report["history"].append(
            dict(
                iteration=n + 1,
                elapsed_s=time.monotonic() - started,
                consensus=con,
                step=step,
                rho=rho,
                rho_next=rho_next,
                penalty_update_reason=penalty_reason,
                multiplier_min=float(np.min(lam)),
                multiplier_max=float(np.max(lam)),
                best_objective=(
                    None if report["best"] is None else report["best"]["objective"]
                ),
            )
        )
        rho = rho_next
        previous_residual = con
        save()
        if stopping:
            report["termination_reason"] = "accepted_stopping_criteria"
            break
    if report["termination_reason"] is None:
        report["termination_reason"] = (
            "time_limit" if remaining(1) <= 0 else "iteration_limit"
        )
    report["status"] = (
        "accepted_full_solution"
        if report["best"] is not None
        else "no_acceptable_full_solution_found"
    )
    save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(json.loads(Path(args.config).read_text()), started=PROCESS_STARTED)
