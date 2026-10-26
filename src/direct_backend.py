"""Physical-period control for the shared UC/AC coordinator.

The discrete domain matches the released adapter. Physical variables, rather
than model inputs, are optimized; no checkpoint or prediction is loaded here.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np

from direct_period import DirectPeriodOracle, DirectPeriodWorker, ROOT, case_identity
from opf_networks import network_spec, nominal_equipment


class DirectPeriodBackend:
    def __init__(self, case, case_path, export_folder, directory, *, deadline):
        self.case = case
        self.case_path = Path(case_path)
        self.export_folder = Path(export_folder)
        self.directory = Path(directory)
        if self.directory.exists():
            raise ValueError("Direct worker directory already exists")
        self.deadline = deadline
        self.worker = None
        self.sessions = []
        self.spec = spec = network_spec(case)
        self.nb, self.ng, self.nl = (
            len(case["bus"]),
            len(case["gen"]),
            len(case["branch"]),
        )
        if (self.nb, self.ng, self.nl) != (
            spec.buses,
            spec.generators,
            spec.lines,
        ) or case["baseMVA"] != 100:
            raise ValueError(
                "Direct control requires the matching released network layout and power base"
            )
        equipment = nominal_equipment(spec)
        self.nominal_gen = equipment["gen"][:, 7]
        self.nominal_line = equipment["branch"][:, 10]
        for rows, field in (
            (case["bus"], "bus"),
            (case["gen"], "idx"),
            (case["branch"], "idx"),
        ):
            if [row[field] for row in rows] != list(range(len(rows))):
                raise ValueError("Original contiguous equipment order required")
            if any(
                row.get("scenario", case.get("scenario", 0)) != case.get("scenario", 0)
                for row in rows
            ):
                raise ValueError("Record scenario IDs disagree with case identity")
        self.gen_bus = np.array([g["bus"] for g in case["gen"]])
        endpoints = np.array(
            [[line["from_bus"], line["to_bus"]] for line in case["branch"]]
        )
        if not np.array_equal(
            self.gen_bus, equipment["gen"][:, 0] - 1
        ) or not np.array_equal(endpoints, equipment["branch"][:, :2] - 1):
            raise ValueError("Equipment connections differ from the nominal network")
        refs = [i for i, bus in enumerate(case["bus"]) if bus["REF"] == 1]
        if len(refs) != 1:
            raise ValueError("Exactly one reference bus required")
        self.ref = refs[0]
        self.ref_gens = np.flatnonzero(
            (self.gen_bus == self.ref) & (self.nominal_gen == 1)
        )
        if not len(self.ref_gens):
            raise ValueError("At least one available reference generator required")
        self.line_status = np.array(
            [line["br_status"] for line in case["branch"]], float
        )
        if not np.isin(self.line_status, [0, 1]).all() or np.any(
            self.line_status > self.nominal_line
        ):
            raise ValueError("Invalid active-line availability")
        neighbors = [set() for _ in case["bus"]]
        for (i, j), active in zip(endpoints, self.line_status):
            if active:
                neighbors[int(i)].add(int(j))
                neighbors[int(j)].add(int(i))
        reached, pending = {self.ref}, [self.ref]
        while pending:
            for neighbor in neighbors[pending.pop()] - reached:
                reached.add(neighbor)
                pending.append(neighbor)
        if len(reached) != self.nb:
            raise ValueError("Disconnected graph is outside the release sampler")
        self.pmin, self.pmax, self.qmin, self.qmax = (
            np.array([g[key] for g in case["gen"]], float)
            for key in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar")
        )
        self.vmin, self.vmax = (
            np.array([b[key] for b in case["bus"]], float)
            for key in ("min_vm_pu", "max_vm_pu")
        )
        bounds = np.r_[self.pmin, self.pmax, self.qmin, self.qmax, self.vmin, self.vmax]
        if (
            not np.isfinite(bounds).all()
            or np.any(self.pmin < 0)
            or np.any(self.pmin > self.pmax)
            or np.any(self.qmin > self.qmax)
            or np.any(self.vmin > self.vmax)
        ):
            raise ValueError("Invalid physical bounds for the direct control")
        ratings = np.array([line["rate_a"] for line in case["branch"]])[
            self.line_status == 1
        ]
        if (
            not np.isfinite(ratings).all()
            or np.any(ratings <= 0)
            or np.any(ratings >= 1e10)
        ):
            raise ValueError(
                "Direct period builder requires finite active ratings in (0,1e10)"
            )
        pd, qd = np.asarray(case["pd"], float), np.asarray(case["qd"], float)
        if (
            pd.shape != (case["T"], self.nb)
            or qd.shape != pd.shape
            or not np.isfinite(pd).all()
            or not np.isfinite(qd).all()
        ):
            raise ValueError("Finite T-by-bus demand arrays required")

    def oracle(self, t, online):
        u = np.asarray(online, float)
        if u.shape != (self.ng,) or not np.isin(u, [0, 1]).all():
            raise ValueError(
                "Commitment must be a binary vector in original generator order"
            )
        if not np.all(u[self.ref_gens] == 1) or np.any(u > self.nominal_gen):
            raise ValueError("Reference generator or nominal availability restriction")
        outages = int(
            np.sum(self.nominal_gen - u) + np.sum(self.nominal_line - self.line_status)
        )
        if not 1 <= outages <= self.spec.maximum_outages:
            raise ValueError("Commitment outside the released additional-outage family")
        if np.any(np.maximum(0.0, self.qmin * u) > self.qmax * u):
            raise ValueError("Empty nonnegative reactive-generation interval")
        return DirectPeriodOracle(self.case, t, u, None)

    def prepare(self, oracle):
        """Restart a closed session before the next solve, charging the global clock."""
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Direct control overall deadline")
        if (
            self.worker is None
            or self.worker.process is None
            or self.worker.process.poll() is not None
        ):
            self.close()
            started = time.monotonic()
            record = dict(
                index=len(self.sessions), started_monotonic=started, status="starting"
            )
            self.sessions.append(record)
            try:
                self.worker = DirectPeriodWorker(
                    self.case_path,
                    self.export_folder,
                    self.directory / ("session" + str(record["index"])),
                    deadline=self.deadline,
                )
                record.update(status="ready", identity=self.worker.ready)
            except (OSError, RuntimeError, ValueError, TimeoutError) as error:
                record.update(status="failed", error=str(error))
                raise
            finally:
                record["startup_s"] = time.monotonic() - started
        if case_identity(oracle.case) != self.worker.case_identity:
            raise ValueError("Direct worker identity mismatch")
        oracle.worker = self.worker

    def initial(self, oracle, horizon, attempt, rng):
        """Horizon dispatch first; later attempts draw bounded physical starts."""
        u = oracle.u
        lower = dict(pg=self.pmin * u, qg=np.maximum(0.0, self.qmin * u), vm=self.vmin)
        upper = dict(pg=self.pmax * u, qg=self.qmax * u, vm=self.vmax)
        if attempt == 0:
            start = {key: (lower[key] + upper[key]) / 2 for key in lower}
            start["pg"] = np.clip(
                horizon["p"][oracle.t] * self.case["baseMVA"], lower["pg"], upper["pg"]
            )
            start["va"] = np.zeros(self.nb)
        else:
            start = {key: rng.uniform(lower[key], upper[key]) for key in lower}
            start["va"] = rng.uniform(-0.1, 0.1, self.nb)
            start["va"][self.ref] = 0.0
        return start

    def domain_record(self, oracle):
        return dict(
            period=oracle.t,
            pattern=oracle.u,
            period_model="direct_ac",
            coordinate_kind="physical",
            original_physical_data_unchanged=True,
        )

    def artifacts(self):
        files = [
            ROOT / "baseline/direct_period/model.jl",
            ROOT / "baseline/direct_period/worker.jl",
            ROOT / "baseline/scripts/companion_adapter.jl",
            *sorted((ROOT / "baseline/companion_matched").glob("*.jl")),
            ROOT / "baseline/environment_direct/Project.toml",
            ROOT / "baseline/environment_direct/Manifest.toml",
        ]
        return {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in files
        }

    def export_record(self):
        path = self.export_folder / "export_manifest.json"
        manifest = json.loads(path.read_text())
        if (
            manifest["source_instance_sha256"]
            != hashlib.sha256(self.case_path.read_bytes()).hexdigest()
        ):
            raise ValueError("Direct physical export belongs to another instance")
        for name, expected in manifest["exported_sha256"].items():
            if (
                Path(name).name != name
                or hashlib.sha256((self.export_folder / name).read_bytes()).hexdigest()
                != expected
            ):
                raise ValueError("Changed or invalid direct export file")
        return dict(
            folder=str(self.export_folder),
            manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            files=manifest["exported_sha256"],
        )

    def close(self):
        if self.worker is not None:
            self.worker.close()
