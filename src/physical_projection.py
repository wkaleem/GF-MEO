"""Sparse original-AC output correction for the corrected-predictor variant.

The distance metric is identity on per-unit powers,
per-unit voltage magnitudes, and radian angles. Raw prediction and empty active
set determine each local solve; there is no global projection guarantee.
Compilation and individual library calls are atomic; deadline checks occur
between calls, and late results cannot be accepted.
"""

import math
import time

import numpy as np
import sympy as sp
from scipy.linalg import null_space
from scipy.sparse import bmat, coo_matrix, eye
from scipy.sparse.linalg import lsqr, splu


class SparseProjection:
    def __init__(self, problem, *, deadline=float("inf")):
        self.deadline = deadline
        self.last_attempt = {}
        self.check_deadline()
        started = time.perf_counter()
        symbols = sp.symbols(problem["variables"])
        self.n = len(symbols)
        lookup = dict(zip(problem["variables"], symbols))
        lookup.update(sin=sp.sin, cos=sp.cos)
        rows = sorted(problem["constraint_details"], key=lambda r: r["kind"] != "eq")
        self.ne = sum(row["kind"] == "eq" for row in rows)
        self.m = len(rows)
        if any(row["kind"] not in ("eq", "ineq") for row in rows):
            raise ValueError("Unknown constraint kind")
        expressions = [sp.sympify(row["residual"], locals=lookup) for row in rows]
        positions = {symbol: i for i, symbol in enumerate(symbols)}
        jr, jc, jvalues, hr, hc, hw, hvalues = [], [], [], [], [], [], []
        for row, expr in enumerate(expressions):
            self.check_deadline()
            present = sorted(expr.free_symbols, key=positions.__getitem__)
            for symbol in present:
                self.check_deadline()
                first = sp.diff(expr, symbol)
                if first == 0:
                    continue
                jr.append(row)
                jc.append(positions[symbol])
                jvalues.append(first)
                for second_symbol in present:
                    second = sp.diff(first, second_symbol)
                    if second != 0:
                        hr.append(positions[symbol])
                        hc.append(positions[second_symbol])
                        hw.append(row)
                        hvalues.append(second)
        self.jr, self.jc = np.array(jr, int), np.array(jc, int)
        self.hr, self.hc, self.hw = (
            np.array(hr, int),
            np.array(hc, int),
            np.array(hw, int),
        )
        self.values = sp.lambdify(
            [symbols], expressions, "numpy", cse=True, docstring_limit=0
        )
        self.check_deadline()
        self.jvalues = sp.lambdify(
            [symbols], jvalues, "numpy", cse=True, docstring_limit=0
        )
        self.check_deadline()
        self.hvalues = sp.lambdify(
            [symbols], hvalues, "numpy", cse=True, docstring_limit=0
        )
        self.compile_seconds = time.perf_counter() - started
        self.check_deadline()

    def check_deadline(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Physical correction deadline")

    def constraints(self, y):
        self.check_deadline()
        return np.asarray(self.values(y), float)

    def jacobian(self, y):
        self.check_deadline()
        return coo_matrix(
            (np.asarray(self.jvalues(y), float), (self.jr, self.jc)),
            shape=(self.m, self.n),
        ).tocsr()

    def hessian(self, y, multipliers):
        self.check_deadline()
        values = np.asarray(self.hvalues(y), float) * multipliers[self.hw]
        return (
            eye(self.n, format="csc")
            + coo_matrix((values, (self.hr, self.hc)), shape=(self.n, self.n)).tocsc()
        )

    def active_rows(self, active):
        self.check_deadline()
        values = np.asarray(active, float)
        if (
            values.ndim != 1
            or not np.isfinite(values).all()
            or np.any(values != np.floor(values))
        ):
            raise ValueError("Invalid active inequality indices")
        active = values.astype(int)
        if (
            len(np.unique(active)) != len(active)
            or np.any(active < 0)
            or np.any(active >= self.m - self.ne)
        ):
            raise ValueError("Invalid active inequality indices")
        return np.r_[np.arange(self.ne), self.ne + active]

    def system(self, y, dual, target, rows):
        J = self.jacobian(y)[rows]
        residual = np.r_[y - target + J.T @ dual, self.constraints(y)[rows]]
        full_dual = np.zeros(self.m)
        full_dual[rows] = dual
        K = bmat([[self.hessian(y, full_dual), J.T], [J, None]], format="csc")
        return residual, K, J

    def solve_active(self, target, y0, active, *, max_iterations=40, tolerance=1e-11):
        if type(max_iterations) is not int or max_iterations < 0:
            raise ValueError("Newton iteration limit must be a nonnegative integer")
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Newton tolerance must be finite and positive")
        target, y = np.asarray(target, float), np.asarray(y0, float).copy()
        if (
            target.shape != (self.n,)
            or y.shape != target.shape
            or not np.isfinite(np.r_[target, y]).all()
        ):
            raise ValueError("Invalid target or physical initial point")
        rows = self.active_rows(active)
        J = self.jacobian(y)[rows]
        dual = lsqr(J.T, target - y, atol=1e-13, btol=1e-13, iter_lim=10 * self.n)[0]
        for iteration in range(max_iterations + 1):
            residual, K, J = self.system(y, dual, target, rows)
            self.last_attempt.update(
                last_iterate=y.tolist(),
                last_multipliers=dual.tolist(),
                active_newton_iterations=iteration,
                last_kkt_residual=(
                    float(np.max(np.abs(residual)))
                    if np.isfinite(residual).all()
                    else None
                ),
            )
            if not np.isfinite(residual).all():
                raise ValueError("Nonfinite KKT residual")
            if np.max(np.abs(residual)) <= tolerance:
                return y, dual, K, J, iteration
            if iteration == max_iterations:
                raise ValueError("Sparse Newton iteration limit")
            step = splu(K).solve(-residual)
            merit = float(residual @ residual)
            for trial in range(20):
                alpha = 0.5**trial
                candidate = y + alpha * step[: self.n]
                candidate_dual = dual + alpha * step[self.n :]
                jc = self.jacobian(candidate)[rows]
                rc = np.r_[
                    candidate - target + jc.T @ candidate_dual,
                    self.constraints(candidate)[rows],
                ]
                if np.isfinite(rc).all() and rc @ rc <= (1 - 1e-4 * alpha) * merit:
                    y, dual = candidate, candidate_dual
                    break
            else:
                raise ValueError("Sparse Newton line search failed")

    def project(
        self,
        target,
        y0,
        initial_active,
        *,
        max_changes=20,
        require_inactive_margin=True,
    ):
        """Local KKT correction; an absent update cap requires a finite deadline.

        An explicit cap is retained for reproducible diagnostics. It counts
        active-set updates, including the final resulting system's evaluation.
        The timed coordinator passes None so the cap does not depend on size.
        """
        if max_changes is None:
            if not np.isfinite(self.deadline):
                raise ValueError("Uncapped correction requires a finite deadline")
        elif type(max_changes) is not int or max_changes < 0:
            raise ValueError(
                "Active-set change limit must be a nonnegative integer or None"
            )
        self.last_attempt = dict(
            max_changes=max_changes,
            changes=[],
            active=[],
            completed_newton_iterations=0,
            last_iterate=None,
            last_multipliers=None,
            last_kkt_residual=None,
            maximum_inequality=None,
        )
        try:
            return self._project(
                target, y0, initial_active, max_changes, require_inactive_margin
            )
        except (ValueError, RuntimeError, TimeoutError, np.linalg.LinAlgError) as error:
            error.diagnostics = dict(self.last_attempt, reason=str(error))
            raise

    def _project(
        self, target, y0, initial_active, max_changes, require_inactive_margin
    ):
        self.active_rows(initial_active)
        active = np.asarray(initial_active, int).copy()
        history = []
        iterations = 0
        y = np.asarray(y0, float)
        visited = {}
        while True:
            self.last_attempt.update(
                active=active.tolist(),
                changes=list(history),
                completed_newton_iterations=iterations,
            )
            self.check_deadline()
            key = tuple(active)
            prior = visited.get(key)
            if prior is not None and np.max(np.abs(y - prior), initial=0.0) <= 1e-9:
                raise ValueError("Active-set cycle without iterate progress")
            visited[key] = y.copy()
            y, dual, K, J, count = self.solve_active(target, y, active)
            iterations += count
            g = self.constraints(y)[self.ne :]
            mu = dual[self.ne :]
            self.last_attempt.update(
                completed_newton_iterations=iterations,
                maximum_inequality=float(np.max(g)) if len(g) else None,
            )
            if len(mu) and np.min(mu) < -1e-9:
                index = int(np.argmin(mu))
                if max_changes is not None and len(history) >= max_changes:
                    raise ValueError("Active-set change limit")
                history.append({"remove": int(active[index])})
                active = np.delete(active, index)
            elif len(g) and np.max(g) > 2e-9:
                index = int(np.argmax(g))
                if index in active:
                    raise ValueError("Violated constraint already active")
                if max_changes is not None and len(history) >= max_changes:
                    raise ValueError("Active-set change limit")
                history.append({"add": index})
                active = np.sort(np.r_[active, index])
            else:
                # Dense rank/curvature checks are deliberate diagnostic checks,
                # outside the sparse Newton iterations, and included in timing.
                singular = np.linalg.svd(J.toarray(), compute_uv=False)
                self.check_deadline()
                tangent = null_space(J.toarray())
                self.check_deadline()
                curvature = np.linalg.eigvalsh(
                    tangent.T @ (K[: self.n, : self.n] @ tangent)
                )
                self.check_deadline()
                inactive = np.setdiff1d(np.arange(len(g)), active)
                if J.shape[0] > self.n or (len(singular) and singular[-1] <= 1e-7):
                    raise ValueError("Dependent active constraints")
                if len(mu) and np.min(mu) <= 1e-7:
                    raise ValueError("Non-strict active multiplier")
                weak_inactive = bool(len(inactive) and np.min(-g[inactive]) <= 1e-7)
                if require_inactive_margin and weak_inactive:
                    raise ValueError("Non-strict inactive slack")
                if len(curvature) and curvature[0] <= 1e-7:
                    raise ValueError("Nonpositive reduced curvature")
                residual = np.r_[
                    y - target + J.T @ dual,
                    self.constraints(y)[self.active_rows(active)],
                ]
                margins = dict(
                    kkt_residual=float(np.max(np.abs(residual))),
                    equality_jacobian_min_singular=(
                        float(singular[-1]) if len(singular) else None
                    ),
                    minimum_active_multiplier=float(np.min(mu)) if len(mu) else None,
                    minimum_inactive_slack=(
                        float(np.min(-g[inactive])) if len(inactive) else None
                    ),
                    minimum_reduced_hessian_eigenvalue=(
                        float(curvature[0]) if len(curvature) else None
                    ),
                    maximum_inequality=float(np.max(g)) if len(g) else None,
                )
                return dict(
                    y=y,
                    dual=dual,
                    K=K,
                    active=active,
                    changes=history,
                    margins=margins,
                    implicit_derivative_available=not weak_inactive,
                    derivative_unavailable_reasons=(
                        ["Non-strict inactive slack"] if weak_inactive else []
                    ),
                    iterations=iterations,
                    min_curvature=margins["minimum_reduced_hessian_eigenvalue"],
                )


def build_problem(case, t, u):
    """Polar original-network equations; eliminate only fixed coordinates.

    Pg,Qg coordinates are per-unit; Vm is per-unit and Va is radians. All
    original power balances, literal nonnegative Q, both-terminal ratings,
    and principal-angle restrictions are included. Temporal constraints are
    enforced by the horizon problem.
    """
    base = case["baseMVA"]
    n = len(case["bus"])
    g = len(case["gen"])
    ids = {b["bus"]: i for i, b in enumerate(case["bus"])}
    free, fixed, constraints, bounds = [], {}, [], {}

    def add(kind, expr):
        constraints.append(dict(kind=kind, residual=expr))

    def variable(name, lo=None, hi=None):
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("Inconsistent physical bounds: " + name)
        if lo is not None and hi is not None and lo == hi:
            fixed[name] = lo
            return repr(float(lo))
        free.append(name)
        bounds[name] = (lo, hi)
        if lo is not None:
            add("ineq", f"({lo})-({name})")
        if hi is not None:
            add("ineq", f"({name})-({hi})")
        return name

    pg = [
        variable(
            f"p{i}", max(0.0, r["min_p_mw"] * u[i]) / base, r["max_p_mw"] * u[i] / base
        )
        for i, r in enumerate(case["gen"])
    ]
    qg = [
        variable(
            f"q{i}",
            max(0.0, r["min_q_mvar"] * u[i]) / base,
            r["max_q_mvar"] * u[i] / base,
        )
        for i, r in enumerate(case["gen"])
    ]
    vm = [
        variable(f"v{i}", b["min_vm_pu"], b["max_vm_pu"])
        for i, b in enumerate(case["bus"])
    ]
    va = [
        variable(f"a{i}", 0.0 if b["REF"] else None, 0.0 if b["REF"] else None)
        for i, b in enumerate(case["bus"])
    ]
    outp = [[] for _ in range(n)]
    outq = [[] for _ in range(n)]
    for e in case["branch"]:
        if not e["br_status"]:
            continue
        i, j = ids[e["from_bus"]], ids[e["to_bus"]]
        ys = 1 / complex(e["r"], e["x"])
        ch = 0.5j * e["b"]
        tap = (e["tap"] or 1.0) * np.exp(1j * np.deg2rad(e["shift"]))
        coeffs = [
            (i, j, (ys + ch) / abs(tap) ** 2, -ys / np.conj(tap)),
            (j, i, ys + ch, -ys / tap),
        ]
        for a, b, yaa, yab in coeffs:
            delta = f"(({va[a]})-({va[b]}))"
            prod = f"({vm[a]})*({vm[b]})"
            p = f"({yaa.real})*({vm[a]})**2+{prod}*(({yab.real})*cos({delta})+({yab.imag})*sin({delta}))"
            q = f"({-yaa.imag})*({vm[a]})**2+{prod}*(({yab.real})*sin({delta})-({yab.imag})*cos({delta}))"
            outp[a].append(f"({p})")
            outq[a].append(f"({q})")
            if e["rate_a"] > 0:
                add("ineq", f'({p})**2+({q})**2-({e["rate_a"]/base})**2')
        delta = f"(({va[i]})-({va[j]}))"
        add("ineq", f"{delta}-({math.pi/2-1e-7})")
        add("ineq", f"-{delta}-({math.pi/2-1e-7})")
    for i, b in enumerate(case["bus"]):
        gens = [k for k, r in enumerate(case["gen"]) if ids[r["bus"]] == i]
        pgen = "+".join(f"({pg[k]})" for k in gens) or "0."
        qgen = "+".join(f"({qg[k]})" for k in gens) or "0."
        p = "+".join(outp[i]) or "0."
        q = "+".join(outq[i]) or "0."
        ratio = case.get("shunt_baseMVA", 100.0) / base
        add(
            "eq",
            f'({pgen})-({case["pd"][t][i]/base})-({p})-({ratio*b["GS"]})*({vm[i]})**2',
        )
        add(
            "eq",
            f'({qgen})-({case["qd"][t][i]/base})-({q})+({ratio*b["BS"]})*({vm[i]})**2',
        )
    problem = dict(
        parameters=["unused"], variables=free, constraint_details=constraints
    )

    def pack(state):
        values = {
            f"{prefix}{i}": float(z) / scale
            for key, prefix, scale in [
                ("pg", "p", base),
                ("qg", "q", base),
                ("vm", "v", 1.0),
                ("va", "a", 1.0),
            ]
            for i, z in enumerate(state[key])
        }
        return np.array([values[k] for k in free])

    def unpack(y):
        values = dict(fixed)
        values.update(zip(free, y))
        return {
            key: np.array([values[f"{prefix}{i}"] * scale for i in range(size)])
            for key, prefix, scale, size in [
                ("pg", "p", base, g),
                ("qg", "q", base, g),
                ("vm", "v", 1.0, n),
                ("va", "a", 1.0, n),
            ]
        }

    return problem, pack, unpack, bounds
