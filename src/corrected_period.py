"""Corrected-output period model, distinct from strict neural-image embedding.

Each evaluation starts its physical correction at the raw neural prediction
with an empty active set. The physical instance and correction metric are
fixed. A scalar implicit adjoint avoids a full neural constraint Jacobian.
"""

import hashlib
import json
import time

import numpy as np
import torch
from scipy.sparse.linalg import splu

from period_oracle import PeriodOracle
from physical_projection import SparseProjection, build_problem
from manuscript_period import period_objective


class CorrectionFailure(FloatingPointError):
    """No independently accepted local distance correction was obtained."""


class GradientUnavailable(FloatingPointError):
    """A feasible correction has no certified smooth implicit derivative."""


class CorrectedPeriodOracle:
    def __init__(
        self,
        case,
        t,
        u,
        radius,
        convention,
        *,
        model_bundle=None,
        optimize_linear_costs=True,
        compiled_cache=None,
        deadline=float("inf"),
    ):
        if convention != "literal_period":
            raise ValueError(
                "Corrected model currently requires literal-period constraints"
            )
        self.raw = PeriodOracle(
            case,
            t,
            u,
            radius,
            convention,
            model_bundle=model_bundle,
            optimize_linear_costs=optimize_linear_costs,
        )
        self.case, self.t, self.u = self.raw.case, t, self.raw.u
        self.z0, self.free = self.raw.z0, self.raw.free
        self.deadline = deadline
        self.cache = None
        problem, self.pack, self.unpack, _ = build_problem(self.case, t, u)
        self.names = problem["variables"]
        self.problem_hash = hashlib.sha256(
            json.dumps(problem, sort_keys=True).encode()
        ).hexdigest()
        cache = {} if compiled_cache is None else compiled_cache
        self.compiled_now = self.problem_hash not in cache
        if self.compiled_now:
            cache[self.problem_hash] = SparseProjection(problem, deadline=deadline)
        self.correction = cache[self.problem_hash]
        self.calls = 0
        self.adjoint_calls = 0
        self.check_deadline()

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Corrected period deadline")

    def domain_record(self):
        return dict(
            self.raw.domain_record(),
            period_model="corrected_output",
            linear_cost_inputs_optimized=self.raw.optimize_linear_costs,
            correction_initialization="raw prediction; empty active set; no prior dispatch",
            correction_metric="identity in packed per-unit Pg/Qg, per-unit Vm, radian Va",
            physical_problem_sha256=self.problem_hash,
            compile_s=self.correction.compile_seconds if self.compiled_now else 0.0,
            cache_hit=not self.compiled_now,
        )

    def evaluate(self, z, derivatives=False):
        self.check_deadline()
        z = np.asarray(z, float)
        if (
            z.shape != self.z0.shape
            or not np.isfinite(z).all()
            or np.max(np.abs(z), initial=0.0) > 1 + 1e-12
        ):
            raise ValueError("Corrected input coordinates outside box")
        if self.cache is not None and np.array_equal(z, self.cache["z"]):
            return self.cache
        tensor = torch.tensor(z, dtype=torch.float64, requires_grad=bool(len(z)))
        self.raw.cache = None
        raw, prediction, _, _ = self.raw.forward(tensor)
        values = dict(
            p=prediction["pg"] / self.case["baseMVA"],
            q=prediction["qg"] / self.case["baseMVA"],
            v=prediction["vm"],
            a=prediction["va"],
        )
        packed = torch.stack([values[name[0]][int(name[1:])] for name in self.names])
        self.check_deadline()
        target = packed.detach().numpy()
        self.correction.deadline = self.deadline
        started = time.monotonic()
        try:
            corrected = self.correction.project(
                target,
                target.copy(),
                [],
                max_changes=None if np.isfinite(self.deadline) else 20,
                require_inactive_margin=False,
            )
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            failure = CorrectionFailure(str(error))
            failure.diagnostics = dict(
                getattr(error, "diagnostics", {}),
                input_coordinates=z.tolist(),
                raw_target=target.tolist(),
                elapsed_s=time.monotonic() - started,
            )
            raise failure from error
        self.check_deadline()
        state = dict(self.unpack(corrected["y"]), uc=self.u.copy())
        result = dict(
            z=z.copy(),
            x=raw.detach().numpy(),
            state=state,
            interface=np.r_[self.u, state["pg"] / self.case["baseMVA"]],
            raw_prediction={
                k: prediction[k].detach().numpy() for k in ("pg", "qg", "vm", "va")
            },
            correction=dict(
                physical_problem_sha256=self.problem_hash,
                target=target,
                physical_variables=corrected["y"],
                active=corrected["active"],
                dual=corrected["dual"],
                changes=corrected["changes"],
                margins=corrected["margins"],
                iterations=corrected["iterations"],
                elapsed_s=time.monotonic() - started,
                implicit_derivative_available=corrected[
                    "implicit_derivative_available"
                ],
                derivative_unavailable_reasons=corrected[
                    "derivative_unavailable_reasons"
                ],
                initialization="raw prediction; empty active set",
            ),
        )
        checked = self.check(result)
        self.check_deadline()
        if not checked["accepted_period"]:
            raise CorrectionFailure(
                "Corrected point failed independent original-AC check"
            )
        for value in list(result.values()) + list(state.values()):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.cache = result
        self._tensor, self._prediction, self._kkt = tensor, packed, corrected
        self.calls += 1
        return result

    def check(self, result):
        checked = self.raw.check(result)
        checked.update(
            neural_state_obtained_by_substitution=False,
            corrected_state_obtained_by_substitution=True,
            scope="single-period corrected-output feasibility; no temporal UC claim",
        )
        return checked

    def gradient(self, entry, target, multipliers, rho, weights):
        self.check_deadline()
        current = self.evaluate(entry["z"])
        if not len(self.z0):
            return np.zeros(0)
        ng = len(self.u)
        coefficient = multipliers + rho * weights * (current["interface"] - target)
        rhs = np.array(
            [
                coefficient[ng + int(name[1:])] if name[0] == "p" else 0.0
                for name in self.names
            ]
        )
        corrected = self._kkt
        if not corrected["implicit_derivative_available"]:
            raise GradientUnavailable(
                "Feasible correction; implicit gradient unavailable: "
                + ", ".join(corrected["derivative_unavailable_reasons"])
            )
        try:
            adjoint = splu(corrected["K"]).solve(
                np.r_[rhs, np.zeros(len(corrected["dual"]))], trans="T"
            )[: len(rhs)]
            value = torch.autograd.grad(
                self._prediction,
                self._tensor,
                grad_outputs=torch.tensor(adjoint, dtype=torch.float64),
                retain_graph=True,
            )[0].numpy()
        except (RuntimeError, np.linalg.LinAlgError) as error:
            raise CorrectionFailure(str(error)) from error
        self.check_deadline()
        if not np.isfinite(value).all():
            raise CorrectionFailure("Nonfinite corrected-input gradient")
        self.adjoint_calls += 1
        return value


def solve_corrected_period(
    oracle,
    *,
    target,
    multipliers,
    rho,
    weights,
    time_limit,
    initial=None,
    previous=None,
    feasibility_only=False,
    maxiter=50,
):
    started = time.monotonic()
    size = 2 * len(oracle.u)
    target, multipliers, weights = (
        np.asarray(v, float) for v in (target, multipliers, weights)
    )
    if any(
        v.shape != (size,) or not np.isfinite(v).all()
        for v in (target, multipliers, weights)
    ):
        raise ValueError("Period interface parameters")
    if not np.isfinite(rho) or rho <= 0 or np.any(weights <= 0):
        raise ValueError("Positive finite penalty and weights required")
    if (
        not np.isfinite(time_limit)
        or time_limit <= 0
        or not isinstance(maxiter, int)
        or maxiter < 1
    ):
        raise ValueError("Positive finite solve limits required")
    z = oracle.z0.copy() if initial is None else np.asarray(initial, float).copy()
    if (
        z.shape != oracle.z0.shape
        or not np.isfinite(z).all()
        or np.max(np.abs(z), initial=0.0) > 1 + 1e-12
    ):
        raise ValueError("Initial period coordinates outside input box")
    oracle.deadline = started + time_limit
    best = None
    checked_points = 0
    iterations = 0
    failures = []
    accepted_steps = []

    def evaluate(z):
        nonlocal best, checked_points
        oracle.check_deadline()
        entry = oracle.evaluate(z)
        check = oracle.check(entry)
        oracle.check_deadline()
        checked_points += 1
        if not check["accepted_period"]:
            raise CorrectionFailure("Unaccepted physical correction")
        objective = period_objective(
            entry["interface"], target, multipliers, rho, weights
        )
        if not np.isfinite(objective):
            raise CorrectionFailure("Nonfinite period objective")
        candidate = dict(entry=entry, check=check, objective=objective)
        if best is None or objective < best["objective"]:
            best = candidate
        return candidate

    status = "iteration limit"
    try:
        if previous is not None:
            evaluate(previous["z"])
        if not (feasibility_only and best is not None):
            current = evaluate(z)
        if feasibility_only:
            status = "accepted corrected feasibility point"
        elif not len(z):
            status = "all input coordinates fixed"
        else:
            for iterations in range(1, maxiter + 1):
                gradient = oracle.gradient(
                    current["entry"], target, multipliers, rho, weights
                )
                norm = np.max(np.abs(gradient))
                if norm < 1e-12:
                    status = "small gradient"
                    break
                scale = 0.1 / norm
                found = False
                for attempt in range(15):
                    proposal = np.clip(
                        current["entry"]["z"] - scale * gradient, -1.0, 1.0
                    )
                    slope = float(gradient @ (proposal - current["entry"]["z"]))
                    if slope >= -1e-16:
                        status = "no projected descent"
                        break
                    try:
                        candidate = evaluate(proposal)
                    except CorrectionFailure as error:
                        failures.append(
                            dict(
                                iteration=iterations,
                                attempt=attempt,
                                reason=str(error),
                                diagnostics=getattr(error, "diagnostics", None),
                            )
                        )
                        scale *= 0.5
                        continue
                    if candidate["objective"] <= current["objective"] + 1e-4 * slope:
                        current = candidate
                        accepted_steps.append(
                            dict(iteration=iterations, objective=current["objective"])
                        )
                        found = True
                        break
                    scale *= 0.5
                if not found:
                    status = "line search stopped"
                    break
    except (TimeoutError, CorrectionFailure, GradientUnavailable) as error:
        status = str(error)
        failures.append(
            dict(
                iteration=iterations,
                phase="initialization" if best is None else "optimization",
                reason=status,
                diagnostics=getattr(error, "diagnostics", None),
            )
        )
    return dict(
        candidate=best,
        solver_status=status,
        iterations=iterations,
        elapsed_s=time.monotonic() - started,
        checked_points=checked_points,
        closest_checked_point=None,
        failures=failures,
        accepted_steps=accepted_steps,
    )
