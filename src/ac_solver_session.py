"""AC subproblem solves with a common time limit."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import signal
import subprocess
import time

import numpy as np
from feasibility import check as check_literal, finite_data
from lagrangian import period_objective

ROOT = Path(__file__).resolve().parents[1]
PREFIX = b"@direct_ac@"
MAX_RECORD_BYTES = 8 * 1024 * 1024


class WorkerError(RuntimeError):
    """The resident worker cannot safely continue this protocol session."""


def _invalid_number(value):
    raise WorkerError("Nonfinite JSON number: " + value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise WorkerError("Duplicate worker JSON key: " + key)
        result[key] = value
    return result


def case_identity(case):
    canonical = json.dumps(case, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()


class DirectPeriodWorker:
    def __init__(
        self,
        case_path,
        export_folder,
        directory,
        *,
        deadline,
        command=None,
        environment=None,
        clock=time.monotonic,
    ):
        self.clock = clock
        self.deadline = float(deadline)
        if not math.isfinite(self.deadline):
            raise ValueError("Finite monotonic deadline required")
        self.process = self.selector = self.log = None
        self.buffer = b""
        self.eof = False
        self.request_id = 0
        self.last_report = None
        self.case_path = Path(case_path).resolve()
        case_bytes = self.case_path.read_bytes()
        self.case_hash = hashlib.sha256(case_bytes).hexdigest()
        self.case_identity = case_identity(json.loads(case_bytes))
        self.ready = None
        try:
            self._remaining(self.deadline)
            directory = Path(directory)
            directory.mkdir(parents=True, exist_ok=False)
            configuration = directory / "worker.json"
            configuration.write_text(
                json.dumps(
                    dict(
                        case=str(self.case_path),
                        export_folder=str(Path(export_folder).resolve()),
                        deadline_unix_s=time.time() + self._remaining(self.deadline),
                        angle_margin=1e-7,
                    ),
                    allow_nan=False,
                )
                + "\n"
            )
            if command is None:
                command = [
                    os.environ.get("JULIA", "julia"),
                    "--startup-file=no",
                    "--project=" + str(ROOT / "baseline/environment_direct"),
                    str(ROOT / "baseline/direct_period/worker.jl"),
                ]
            env = os.environ.copy()
            env.update(environment or {})
            env.update(
                JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1"
            )
            self.log = (directory / "worker.log").open("xb", buffering=0)
            self.process = subprocess.Popen(
                [*command, str(configuration)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                bufsize=0,
                env=env,
                start_new_session=True,
            )
            os.set_blocking(self.process.stdin.fileno(), False)
            os.set_blocking(self.process.stdout.fileno(), False)
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.process.stdout, selectors.EVENT_READ)
            ready = self._receive(self.deadline)
            if (
                ready.get("kind") != "ready"
                or ready.get("case_sha256") != self.case_hash
                or ready.get("coordinate_kind") != "physical"
                or ready.get("checkpoint_loaded") is not False
                or ready.get("angle_margin") != 1e-7
            ):
                raise WorkerError("Worker identity or physical convention mismatch")
            self._remaining(self.deadline)
            self.ready = ready
        except BaseException:
            self.close()
            raise

    def _remaining(self, deadline):
        remaining = min(deadline, self.deadline) - self.clock()
        if remaining <= 0:
            raise TimeoutError("Direct-AC worker deadline reached")
        return remaining

    def _read_chunk(self):
        try:
            chunk = os.read(self.process.stdout.fileno(), 65536)
        except BlockingIOError:
            return
        if not chunk:
            self.eof = True
            self.selector.unregister(self.process.stdout)
        else:
            self.buffer += chunk
            if len(self.buffer) > MAX_RECORD_BYTES:
                raise WorkerError("Worker output exceeds the record limit")

    def _receive(self, deadline):
        while True:
            self._remaining(deadline)
            if b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                if not line.startswith(PREFIX):
                    self.log.write(line + b"\n")
                    continue
                try:
                    record = json.loads(
                        line[len(PREFIX) :],
                        parse_constant=_invalid_number,
                        object_pairs_hook=_unique_object,
                    )
                except (ValueError, UnicodeError) as error:
                    raise WorkerError("Malformed worker JSON") from error
                if not isinstance(record, dict):
                    raise WorkerError("Worker record is not an object")
                if not finite_data(record):
                    raise WorkerError("Nonfinite worker record")
                self._remaining(deadline)
                return record
            if self.eof:
                raise WorkerError("Worker disconnected or returned a truncated record")
            for _, _ in self.selector.select(self._remaining(deadline)):
                self._read_chunk()

    def _send(self, record, deadline):
        message = memoryview((json.dumps(record, allow_nan=False) + "\n").encode())
        self.selector.register(self.process.stdin, selectors.EVENT_WRITE)
        try:
            while message:
                for key, _ in self.selector.select(self._remaining(deadline)):
                    if key.fileobj is self.process.stdout:
                        self._read_chunk()
                    else:
                        try:
                            sent = os.write(self.process.stdin.fileno(), message)
                        except BlockingIOError:
                            continue
                        if sent == 0:
                            raise WorkerError("Worker input disconnected")
                        message = message[sent:]
                if self.eof:
                    raise WorkerError("Worker disconnected during a request")
            self._remaining(deadline)
        finally:
            self.selector.unregister(self.process.stdin)

    def solve(self, request, observe, *, deadline):
        if self.process is None or self.process.poll() is not None:
            self.close()
            raise WorkerError("Worker session is closed")
        self.request_id += 1
        identifier = self.request_id
        self.last_report = None
        sequence = 0
        try:
            request = dict(
                request,
                operation="solve",
                id=identifier,
                seconds=self._remaining(deadline),
            )
            self._send(request, deadline)
            while True:
                record = self._receive(deadline)
                if type(record.get("id")) is not int or record["id"] != identifier:
                    raise WorkerError("Worker request ID mismatch")
                kind = record.get("kind")
                if kind == "candidate":
                    sequence += 1
                    if (
                        type(record.get("sequence")) is not int
                        or record["sequence"] != sequence
                    ):
                        raise WorkerError("Worker candidate sequence mismatch")
                    metadata = record.get("metadata", {})
                    if not isinstance(metadata, dict):
                        raise WorkerError("Malformed candidate metadata")
                    keep_going = observe(record["physical_coordinates"], metadata)
                    if type(keep_going) is not bool:
                        raise WorkerError("Candidate observer must return a Boolean")
                    self._remaining(deadline)
                    self._send(
                        dict(
                            id=identifier, sequence=sequence, **{"continue": keep_going}
                        ),
                        deadline,
                    )
                elif kind == "done":
                    report = record["report"]
                    if not isinstance(report, dict) or not isinstance(
                        report.get("status"), str
                    ):
                        raise WorkerError("Malformed period report")
                    self.last_report = report
                    if report.get("observation_error") is not None:
                        self.close()
                    return report
                elif kind == "error":
                    raise WorkerError("Worker error: " + str(record.get("error")))
                else:
                    raise WorkerError("Unexpected worker message")
        except BaseException:
            self.close()
            raise

    def close(self):
        # Do not send close into a stream that may be waiting for an ACK.
        if self.process is not None:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.process.wait()
            for stream in (self.process.stdin, self.process.stdout):
                stream.close()
            self.process = None
        if self.selector is not None:
            self.selector.close()
            self.selector = None
        if self.log is not None:
            self.log.close()
            self.log = None

    def __enter__(self):
        return self

    def __exit__(self, *exception):
        self.close()


class DirectPeriodOracle:
    def __init__(self, case, t, online, worker):
        if worker is not None and case_identity(case) != worker.case_identity:
            raise ValueError(
                "Period oracle and worker have different physical instances"
            )
        if type(t) is not int or not 0 <= t < case["T"]:
            raise ValueError("Invalid zero-based period")
        self.case = copy.deepcopy(case)
        self.t = t
        self.u = np.array(online, dtype=float, copy=True)
        if self.u.shape != (len(case["gen"]),) or not np.isin(self.u, [0.0, 1.0]).all():
            raise ValueError("Invalid online indicators")
        for name in (
            "ramp_up_mw",
            "ramp_down_mw",
            "startup_ramp_mw",
            "shutdown_ramp_mw",
        ):
            limits = np.asarray(case[name], float)
            if (
                limits.shape != self.u.shape
                or not np.isfinite(limits).all()
                or np.any(limits < 0)
            ):
                raise ValueError("Invalid original ramp limits")
        self.u.setflags(write=False)
        self.worker = worker

    def evaluate(self, coordinates):
        state = {"uc": self.u}
        if set(coordinates) != {"pg", "qg", "vm", "va"}:
            raise ValueError("Physical coordinates require pg, qg, vm and va")
        for name in ("pg", "qg", "vm", "va"):
            try:
                value = np.array(coordinates[name], dtype=float, copy=True)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError("Invalid physical coordinates: " + name) from error
            size = len(self.u) if name in ("pg", "qg") else len(self.case["bus"])
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError("Invalid physical coordinates: " + name)
            value.setflags(write=False)
            state[name] = value
        interface = np.r_[self.u, state["pg"] / self.case["baseMVA"]]
        interface.setflags(write=False)
        return dict(
            state=state,
            interface=interface,
            coordinate_kind="physical",
            physical_coordinates={name: state[name] for name in coordinates},
        )

    def check(self, entry):
        state = entry["state"]
        if not np.array_equal(state["uc"], self.u):
            raise ValueError("Physical candidate changed its online indicators")
        case = copy.deepcopy(self.case)
        case.update(
            T=1,
            pd=[case["pd"][self.t]],
            qd=[case["qd"][self.t]],
            initial_on=self.u.tolist(),
            initial_pg_mw=state["pg"].tolist(),
            initial_forced_on=self.u.tolist(),
            initial_forced_off=(1 - self.u).tolist(),
            min_up=[1] * len(self.u),
            min_down=[1] * len(self.u),
        )
        solution = {name: [state[name].tolist()] for name in ("pg", "qg", "vm", "va")}
        solution.update(
            u=[self.u.tolist()], su=[[0] * len(self.u)], sd=[[0] * len(self.u)]
        )
        checked = check_literal(case, solution)
        ids = {bus["bus"]: i for i, bus in enumerate(case["bus"])}
        excess = max(
            [0.0]
            + [
                abs(
                    state["va"][ids[line["from_bus"]]]
                    - state["va"][ids[line["to_bus"]]]
                )
                - (math.pi / 2 - 1e-7)
                for line in case["branch"]
                if line["br_status"]
            ]
        )
        return dict(
            accepted_period=bool(checked["accepted"] and excess <= 1e-6),
            ac_constraints=checked,
            principal_angle_margin_violation=float(excess),
            scope="physical period constraints only; no temporal UC claim",
        )


def solve_direct_period(
    oracle,
    target,
    multipliers,
    rho,
    weights,
    time_limit,
    *,
    initial=None,
    previous=None,
    feasibility_only=False,
):
    """Return only points checked in time; the coordinator also retains its prior.

    If the budget expires before a previous point can be rechecked, this call may
    return no candidate. It must not erase the coordinator's earlier accepted
    entry or relabel an earlier acceptance as a new timely observation.
    """
    started = oracle.worker.clock()
    deadline = min(oracle.worker.deadline, started + time_limit)
    size = 2 * len(oracle.u)
    target, multipliers, weights = (
        np.asarray(v, float) for v in (target, multipliers, weights)
    )
    if any(
        v.shape != (size,) or not np.isfinite(v).all()
        for v in (target, multipliers, weights)
    ):
        raise ValueError("Period interface parameters")
    if not math.isfinite(rho) or rho <= 0 or np.any(weights <= 0):
        raise ValueError("Positive finite penalty and weights required")
    if (
        not math.isfinite(time_limit)
        or time_limit <= 0
        or type(feasibility_only) is not bool
    ):
        raise ValueError("Invalid solve budget or feasibility mode")
    best = None
    checked_points = 0
    failures = []
    report = {}

    def observe(coordinates, metadata):
        nonlocal best, checked_points
        oracle.worker._remaining(deadline)
        entry = oracle.evaluate(coordinates)
        check = oracle.check(entry)
        objective = float(
            period_objective(entry["interface"], target, multipliers, rho, weights)
        )
        if not math.isfinite(objective):
            raise ValueError("Nonfinite period objective")
        observation = dict(metadata)
        accepted_time = oracle.worker.clock()
        if accepted_time >= deadline:
            raise TimeoutError("Direct-AC candidate validation reached the deadline")
        checked_points += 1
        if check["accepted_period"] and (best is None or objective < best["objective"]):
            best = dict(
                entry=entry,
                check=check,
                objective=objective,
                observation=observation,
                accepted_at_s=accepted_time - started,
            )
        return not (feasibility_only and best is not None)

    status = "not_started"
    try:
        if previous is not None:
            observe(previous["physical_coordinates"], {"phase": "previous"})
        if feasibility_only and best is not None:
            status = "accepted previous physical point"
        else:
            start = (
                initial
                if initial is not None
                else (None if previous is None else previous["physical_coordinates"])
            )
            if start is not None:
                start = {
                    k: v.tolist()
                    for k, v in oracle.evaluate(start)["physical_coordinates"].items()
                }
            request = dict(
                period=oracle.t + 1,
                online=oracle.u.tolist(),
                target=target.tolist(),
                multipliers=multipliers.tolist(),
                penalty=float(rho),
                weights=weights.tolist(),
                physical_start=start,
                feasibility_only=feasibility_only,
            )
            report = oracle.worker.solve(request, observe, deadline=deadline)
            status = report["status"]
            if report.get("observation_error") is not None:
                failures.append(dict(reason=report["observation_error"]))
    except (
        TimeoutError,
        WorkerError,
        OSError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        oracle.worker.close()
        status = str(error)
        failures.append(dict(reason=status))
    return dict(
        candidate=best,
        solver_status=status,
        iterations=report.get("iterations"),
        elapsed_s=oracle.worker.clock() - started,
        checked_points=checked_points,
        failures=failures,
        worker_report=report,
    )
