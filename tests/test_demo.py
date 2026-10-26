"""Artifact integrity, isolated setup and solver failure reporting."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import runpy
from types import SimpleNamespace
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    value = module("setup_demo")
    monkeypatch.setattr(value, "ROOT", tmp_path)
    return value


def test_download_checks_content_and_reuses_verified_artifact(
    setup, tmp_path, monkeypatch
):
    raw = b"checkpoint"
    record = dict(
        url="https://example.invalid/model",
        bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
    )
    calls = []
    monkeypatch.setattr(
        setup.urllib.request,
        "urlopen",
        lambda *a, **k: calls.append(a) or io.BytesIO(raw),
    )
    destination = tmp_path / "weights/last.pt"
    assert setup.download(record, destination) == raw
    assert setup.download(record, destination) == raw
    assert len(calls) == 1
    destination.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="identity"):
        setup.download(record, destination)
    assert destination.read_bytes() == b"corrupt"


def test_download_without_certifi_and_with_bad_hash(setup, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "certifi", None)
    monkeypatch.setattr(
        setup.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(b"x")
    )
    with pytest.raises(ValueError, match="identity"):
        setup.download(
            dict(url="https://example.invalid", bytes=1, sha256="bad"), tmp_path / "bad"
        )
    assert not (tmp_path / "bad").exists()


def test_configured_environment_preserves_symlink_and_backend(
    setup, tmp_path, monkeypatch
):
    target = tmp_path / "system-python"
    target.touch()
    interpreter = tmp_path / "venv-python"
    interpreter.symlink_to(target)
    calls = []
    progress = []
    monkeypatch.setattr(
        setup.subprocess, "run", lambda cmd, **kw: calls.append((cmd, kw))
    )
    assert setup.environment(
        str(interpreter), lambda n, s: progress.append((n, s))
    ) == str(interpreter)
    assert calls[0][0][0] == str(interpreter)
    assert calls[0][1]["env"]["MPLBACKEND"] == "Agg"
    assert [n for n, s in progress] == [0, 1, 2]


def test_automatic_install_is_separate_and_checked(setup, tmp_path, monkeypatch):
    monkeypatch.delenv("GFMO_DEMO_PYTHON", raising=False)
    monkeypatch.setattr(setup.sys, "version_info", (3, 11))
    monkeypatch.setattr(setup.platform, "system", lambda: "Linux")
    monkeypatch.setattr(setup.platform, "machine", lambda: "x86_64")
    calls = []
    monkeypatch.setattr(setup.subprocess, "run", lambda cmd, **kw: calls.append(cmd))
    interpreter = setup.environment()
    assert calls[0][1:3] == ["-m", "venv"]
    assert [interpreter, "-m", "pip", "check"] in calls
    Path(interpreter).parent.mkdir(parents=True)
    Path(interpreter).touch()
    calls.clear()
    setup.environment()
    assert not any("venv" in cmd for cmd in calls)


@pytest.mark.parametrize(
    "version,system,machine",
    [
        ((3, 10), "Linux", "x86_64"),
        ((3, 11), "Darwin", "arm64"),
        ((3, 11), "Linux", "arm64"),
    ],
)
def test_unsupported_automatic_install_fails_clearly(
    setup, monkeypatch, version, system, machine
):
    monkeypatch.delenv("GFMO_DEMO_PYTHON", raising=False)
    monkeypatch.setattr(setup.sys, "version_info", version)
    monkeypatch.setattr(setup.platform, "system", lambda: system)
    monkeypatch.setattr(setup.platform, "machine", lambda: machine)
    with pytest.raises(RuntimeError, match="Python 3.11"):
        setup.environment()


def archive(entries):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as z:
        for key, value in entries.items():
            z.writestr(key, value)
    return stream.getvalue()


def test_prepare_matches_size_seed_and_rejects_modified_source(
    setup, tmp_path, monkeypatch
):
    prefix = "gridfm-graphkit-4e7fbcf6aa33c09b5ec9f15e5952dbbff004b74e/"
    raw = archive(
        {
            prefix + "gridfm_graphkit/": "",
            prefix + "gridfm_graphkit/example.py": "x=1",
            prefix + "LICENSE": "license",
            prefix + "README": "ignore",
            "elsewhere": "ignore",
        }
    )
    calls = []

    def download(record, destination):
        calls.append(str(destination))
        return raw

    monkeypatch.setattr(setup, "download", download)
    (tmp_path / "src").mkdir()
    records = [
        dict(path=f"case57_ieee/base/seed0/{name}")
        for name in ["config.yml", "last.pt", "normalizer_stats.pt"]
    ]
    inventory = tmp_path / "src/genco_opf_release.json"
    inventory.write_text(json.dumps(dict(records=records)))
    setup.prepare()
    setup.prepare()
    assert sum(path.endswith("last.pt") for path in calls) == 2
    source = (
        tmp_path / "vendor/gridfm-gridfm-graphkit-4e7fbcf/gridfm_graphkit/example.py"
    )
    source.write_text("changed")
    with pytest.raises(ValueError, match="differs"):
        setup.prepare()
    source.write_text("x=1")
    inventory.write_text('{"records":[]}')
    with pytest.raises(ValueError, match="Incomplete"):
        setup.prepare()
    monkeypatch.setattr(
        setup, "download", lambda *a: archive({prefix + "../escape": "bad"})
    )
    with pytest.raises(ValueError, match="Unsafe"):
        setup.prepare()


@pytest.mark.parametrize("size,seed", [("large", 0), ("base", True), ("small", 2)])
def test_invalid_checkpoint_selection(setup, size, seed):
    with pytest.raises(ValueError):
        setup.prepare(size, seed)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    value = module("run_demo")
    monkeypatch.setattr(value, "ROOT", tmp_path)
    (tmp_path / "demo_config.json").write_text((ROOT / "demo_config.json").read_text())
    (tmp_path / "data").mkdir()
    (tmp_path / "data/ieee57_uc.json").write_text("{}")
    monkeypatch.setenv("GFMO_DEMO_PROGRESS_FILE", str(tmp_path / "progress.json"))
    return value


def launch(runner, monkeypatch, report=None, returncode=0, timeout=False):
    def run(command, **kwargs):
        assert kwargs["env"]["MPLBACKEND"] == "Agg"
        assert kwargs["env"]["OMP_NUM_THREADS"] == "1"
        config = json.loads(Path(command[-1]).read_text())
        if report is not None:
            (runner.ROOT / config["output"]).write_text(json.dumps(report))
        if timeout:
            raise subprocess.TimeoutExpired(command, 360)
        return SimpleNamespace(returncode=returncode)

    monkeypatch.setattr(runner.subprocess, "run", run)


@pytest.mark.parametrize("returncode,report", [(1, None), (0, None)])
def test_execution_failure_is_not_reported_as_infeasibility(
    runner, monkeypatch, returncode, report
):
    launch(runner, monkeypatch, report, returncode)
    with pytest.raises(RuntimeError):
        runner.run_demo()
    folder = Path(json.loads((runner.ROOT / "latest_run.json").read_text())["folder"])
    assert json.loads((folder / "summary.json").read_text())["accepted"] is False


def test_time_limit_without_solution_is_explicit(runner, monkeypatch):
    launch(runner, monkeypatch, timeout=True)
    result = runner.run_demo()
    assert result["process_timeout"] and not result["accepted"]
    assert result["first_feasible_s"] is None


def test_no_incumbent_and_repeated_runs_remain_separate(runner, monkeypatch):
    launch(runner, monkeypatch, {"status": "no_acceptable_full_solution_found"})
    first = runner.run_demo()
    second = runner.run_demo()
    assert not first["accepted"] and first["run_directory"] != second["run_directory"]


def accepted_report(runner):
    best = dict(
        horizon=dict(u=[[1]], su=[[0]], sd=[[0]]),
        periods=[dict(state=dict(pg=[1], qg=[0], vm=[1], va=[0]))],
        objective=1,
        accepted_at_s=3,
    )
    return dict(
        status="accepted_full_solution",
        best=best,
        first_accepted_at_s=3,
        case_sha256=hashlib.sha256(
            (runner.ROOT / "data/ieee57_uc.json").read_bytes()
        ).hexdigest(),
    )


def test_saved_witness_is_independently_checked(runner, monkeypatch):
    report = accepted_report(runner)
    seen = []

    def check(case, witness):
        seen.append(witness)
        return dict(accepted=True)

    monkeypatch.setitem(sys.modules, "literal_check", SimpleNamespace(check=check))
    launch(runner, monkeypatch, report)
    result = runner.run_demo()
    assert result["accepted"] and len(seen) == 1
    assert (
        json.loads((Path(result["run_directory"]) / "witness.json").read_text())
        == seen[0]
    )


def test_changed_case_is_rejected(runner, monkeypatch):
    report = accepted_report(runner)
    report["case_sha256"] = "wrong"
    monkeypatch.setitem(
        sys.modules,
        "literal_check",
        SimpleNamespace(check=lambda *a: pytest.fail("Changed case")),
    )
    launch(runner, monkeypatch, report)
    with pytest.raises(ValueError, match="Case changed"):
        runner.run_demo()


@pytest.mark.parametrize(
    "args",
    [
        {"size": "bad"},
        {"seed": True},
        {"seed": 2},
        {"seconds": 0},
        {"seconds": 3601},
        {"seconds": "300"},
    ],
)
def test_invalid_run_arguments(runner, args):
    with pytest.raises(ValueError):
        runner.run_demo(**args)


@pytest.mark.parametrize(
    "script,args",
    [("run_demo.py", ["--seconds", "0"]), ("setup_demo.py", ["--seed", "2"])],
)
def test_command_line_rejects_invalid_inputs_before_work(script, args, monkeypatch):
    monkeypatch.setattr(sys, "argv", [script] + args)
    with pytest.raises(ValueError if script == "run_demo.py" else SystemExit):
        runpy.run_path(str(ROOT / script), run_name="__main__")


def test_command_line_without_notebook_marker(runner, monkeypatch):
    monkeypatch.delenv("GFMO_DEMO_PROGRESS_FILE")
    launch(runner, monkeypatch, {"status": "no_acceptable_full_solution_found"})
    assert not runner.run_demo()["accepted"]


def test_result_plots_require_a_solution_and_show_real_report_data(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    view = module("demo_view")
    report = tmp_path / "report.json"
    destination = tmp_path / "result.svg"
    report.write_text('{"best":null}')
    assert not view.render(report, destination) and not destination.exists()
    case = json.loads((ROOT / "data/ieee57_uc.json").read_text())
    T = case["T"]
    G = len(case["gen"])
    B = len(case["bus"])
    state = dict(pg=[0.0] * G, qg=[0.0] * G, vm=[1.0] * B, va=[0.0] * B)
    best = dict(
        horizon=dict(u=[[1] * G] * T, su=[[0] * G] * T, sd=[[0] * G] * T),
        periods=[dict(raw_prediction=state, state=state)] * T,
    )
    value = dict(
        best=best,
        configuration=dict(case="data/ieee57_uc.json"),
        accepted_history=[dict(elapsed_s=2, best_objective=10)],
    )
    report.write_text(json.dumps(value))
    assert view.render(report, destination)
    svg = destination.read_text()
    for text in [
        "Best feasible cost",
        "Commitment schedule",
        "Maximum AC balance residual",
        "GENCO prediction",
        "Corrected output",
    ]:
        assert text in svg
    monkeypatch.setattr(sys, "argv", ["demo_view.py", str(report), str(destination)])
    runpy.run_path(str(ROOT / "demo_view.py"), run_name="__main__")
