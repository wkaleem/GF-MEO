"""Solve the IEEE57 example and check the UC–ACOPF schedule."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run_example(size="base", seed=0, seconds=300):
    if size not in ("small", "base") or type(seed) is not int or seed not in (0, 1):
        raise ValueError("Invalid model configuration")
    if not isinstance(seconds, (int, float)) or not 0 < seconds <= 3600:
        raise ValueError("Choose a positive budget no larger than 3600 seconds")
    folder = (
        ROOT
        / "runs"
        / (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "_"
            + uuid.uuid4().hex[:8]
        )
    )
    folder.mkdir(parents=True)
    config = json.loads((ROOT / "config/parameters.json").read_text())
    config.update(
        model_size=size,
        training_seed=seed,
        seconds=float(seconds),
        output=str((folder / "solver.json").relative_to(ROOT)),
    )
    for key in (
        "initialization_seconds",
        "period_seconds",
        "horizon_seconds",
        "completion_seconds",
    ):
        config[key] = float(seconds)
    config_path = folder / "config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    if os.environ.get("GFMO_DEMO_PROGRESS_FILE"):
        Path(os.environ["GFMO_DEMO_PROGRESS_FILE"]).write_text(
            json.dumps({"folder": str(folder)})
        )
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        env[key] = "1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    start = time.monotonic()
    timed_out = False
    with (folder / "solver.log").open("w") as log:
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "src/decomposition.py",
                    "--config",
                    str(config_path),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=seconds + 60,
                check=False,
            )
            returncode = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            returncode = None
    elapsed = time.monotonic() - start
    report_path = folder / "solver.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    summary = dict(
        run_directory=str(folder),
        process_returncode=returncode,
        process_timeout=timed_out,
        process_elapsed_s=elapsed,
        solver_status=report.get("status", "no_report"),
        first_feasible_s=report.get("first_accepted_at_s"),
        accepted=False,
        timing="First feasible: solver-entry clock including imports, model loading, initialization, correction, and UC checks. Process elapsed also includes interpreter startup and shutdown. Installation/download excluded.",
        configuration=config,
        global_optimality_claim=False,
    )
    best = report.get("best")
    if best is not None:
        sys.path.insert(0, str(ROOT / "src"))
        from feasibility import check

        case_path = ROOT / config["case"]
        if hashlib.sha256(case_path.read_bytes()).hexdigest() != report["case_sha256"]:
            raise ValueError("Case changed during execution")
        case = json.loads(case_path.read_text())
        witness = {key: best["horizon"][key] for key in ("u", "su", "sd")}
        witness.update(
            {
                key: [p["state"][key] for p in best["periods"]]
                for key in ("pg", "qg", "vm", "va")
            }
        )
        checked = check(case, witness)
        summary.update(
            accepted=checked["accepted"],
            ac_constraints=checked,
            objective=best["objective"],
            accepted_at_s=best["accepted_at_s"],
        )
        (folder / "witness.json").write_text(json.dumps(witness, indent=2) + "\n")
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n"
    )
    (ROOT / "latest_run.json").write_text(
        json.dumps(dict(folder=str(folder)), indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    if returncode not in (0, None):
        raise RuntimeError("Solver execution failed. See " + str(folder / "solver.log"))
    if not report and not timed_out:
        raise RuntimeError(
            "Solver returned no report. See " + str(folder / "solver.log")
        )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "base"), default="base")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    parser.add_argument("--seconds", type=float, default=300)
    args = parser.parse_args()
    run_example(args.size, args.seed, args.seconds)
