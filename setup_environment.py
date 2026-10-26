"""CPU environment and GENCO model setup."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import ssl
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def environment(configured=None, progress=None):
    """Use the requested interpreter or install an isolated Linux CPU runtime."""
    update = progress if progress is not None else lambda step, label: print(label)
    update(0, "CPU environment")
    configured = configured or os.environ.get("GFMO_DEMO_PYTHON")
    if configured:
        # Preserve a virtual environment's interpreter symlink.
        python = os.path.abspath(os.path.expanduser(configured))
    else:
        if (
            sys.version_info[:2] != (3, 11)
            or platform.system() != "Linux"
            or platform.machine() != "x86_64"
        ):
            raise RuntimeError(
                "Automatic setup requires Python 3.11 on Linux x86_64; see README."
            )
        python = str(ROOT.absolute() / ".venv/bin/python")
        if not Path(python).exists():
            subprocess.run(
                [sys.executable, "-m", "venv", str(ROOT / ".venv")], check=True
            )
    update(1, "CPU dependencies")
    commands = (
        []
        if configured
        else [
            [
                python,
                "-m",
                "pip",
                "install",
                "torch==2.8.0",
                "--index-url",
                "https://download.pytorch.org/whl/cpu",
            ],
            [
                python,
                "-m",
                "pip",
                "install",
                "--no-index",
                "torch-scatter==2.1.2+pt28cpu",
                "-f",
                "https://data.pyg.org/whl/torch-2.8.0+cpu.html",
            ],
            [
                python,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "-r",
                str(ROOT / "requirements-linux.txt"),
            ],
            [python, "-m", "pip", "check"],
        ]
    )
    commands.append(
        [
            python,
            "-c",
            "import torch, torch_geometric, torch_scatter, pyscipopt, scipy, sympy; print(torch.__version__)",
        ]
    )
    env = dict(os.environ, MPLBACKEND="Agg")
    with (ROOT / "setup.log").open("a") as log:
        for command in commands:
            subprocess.run(
                command, check=True, stdout=log, stderr=subprocess.STDOUT, env=env
            )
    update(2, "Runtime ready")
    return python


def digest(data):
    return hashlib.sha256(data).hexdigest()


def download(record, destination):
    destination = Path(destination)
    if destination.exists():
        data = destination.read_bytes()
    else:
        try:
            import certifi

            context = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            context = ssl.create_default_context()
        with urllib.request.urlopen(
            record["url"], context=context, timeout=180
        ) as response:
            data = response.read()
    if len(data) != record["bytes"] or digest(data) != record["sha256"]:
        raise ValueError("Artifact identity differs: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.write_bytes(data)
        temporary.replace(destination)
    return data


def prepare(size="base", seed=0):
    if size not in ("small", "base") or type(seed) is not int or seed not in (0, 1):
        raise ValueError("Choose small/base and seed 0/1")
    record = dict(
        url="https://codeload.github.com/gridfm/gridfm-graphkit/zip/4e7fbcf6aa33c09b5ec9f15e5952dbbff004b74e",
        bytes=4514089,
        sha256="14ac0d29e2c59c5abbd346f024af986b13caafc71cacf59f6451c3f7ebec47e0",
    )
    raw = download(record, ROOT / ".downloads/graphkit.zip")
    prefix = "gridfm-graphkit-4e7fbcf6aa33c09b5ec9f15e5952dbbff004b74e/"
    vendor = ROOT / "vendor/gridfm-gridfm-graphkit-4e7fbcf"
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for entry in archive.infolist():
            if entry.is_dir() or not entry.filename.startswith(prefix):
                continue
            relative = PurePosixPath(entry.filename[len(prefix) :])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe archive path")
            if not (
                str(relative).startswith("gridfm_graphkit/")
                or str(relative) == "LICENSE"
            ):
                continue
            target = vendor / str(relative)
            content = archive.read(entry)
            if target.exists() and target.read_bytes() != content:
                raise ValueError("Existing upstream source differs: " + str(relative))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    nominal = dict(
        url="https://raw.githubusercontent.com/gridfm/gridfm-datakit/316c4afeb1b1d048b6ccaa6135fbc54e00549f64/gridfm_datakit/grids/pglib_opf_case57_ieee_corrected.m",
        bytes=12622,
        sha256="50da73ecb09c02d77f73eaf2f3ad41dfa77a1ec10f85ef4cb07e2f683d42e842",
    )
    download(nominal, ROOT / "sources/distribution_audit/case57_nominal_repro.m")
    inventory = json.loads((ROOT / "src/genco_models.json").read_text())
    selection = f"case57_ieee/{size}/seed{seed}/"
    records = [r for r in inventory["records"] if r["path"].startswith(selection)]
    if len(records) != 3:
        raise ValueError("Incomplete checkpoint bundle")
    for record in records:
        download(record, ROOT / "models/opf_inventory" / record["path"])
    print(f"Verified Graphkit, IEEE 57 network, and GENCO {size}/seed{seed}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", choices=("small", "base"), default="base")
    parser.add_argument("--seed", type=int, choices=(0, 1), default=0)
    arguments = parser.parse_args()
    prepare(arguments.size, arguments.seed)
