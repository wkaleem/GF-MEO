"""Network data for the released GENCO models."""

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np


@dataclass(frozen=True)
class NetworkSpec:
    name: str
    display: str
    buses: int
    generators: int
    lines: int
    maximum_outages: int
    global_load_range: tuple[float, float]
    nominal_path: str


SPECS = {
    "case14_ieee": NetworkSpec(
        "case14_ieee",
        "IEEE14",
        14,
        5,
        20,
        2,
        (0.72, 1.2),
        "sources/distribution_audit/case14_nominal_repro.m",
    ),
    "case30_ieee": NetworkSpec(
        "case30_ieee",
        "IEEE30",
        30,
        6,
        41,
        5,
        (0.66, 1.1),
        "sources/distribution_audit/case30_nominal_repro.m",
    ),
    "case57_ieee": NetworkSpec(
        "case57_ieee",
        "IEEE57",
        57,
        7,
        80,
        5,
        (0.66, 1.1),
        "sources/distribution_audit/case57_nominal_repro.m",
    ),
    "case118_ieee": NetworkSpec(
        "case118_ieee",
        "IEEE118",
        118,
        54,
        186,
        10,
        (0.78, 1.3),
        "sources/distribution_audit/case118_nominal_repro.m",
    ),
    "case500_goc": NetworkSpec(
        "case500_goc",
        "GOC500",
        500,
        224,
        733,
        10,
        (0.72, 1.2),
        "sources/distribution_audit/case500_nominal_repro.m",
    ),
    "case2000_goc": NetworkSpec(
        "case2000_goc",
        "GOC2000",
        2000,
        384,
        3639,
        10,
        (0.84, 1.4),
        "sources/distribution_audit/case2000_nominal_repro.m",
    ),
}


def nominal_equipment(spec):
    """Original ordered generator/line records, before sampled perturbations."""
    root = Path(__file__).resolve().parents[1]
    text = (root / spec.nominal_path).read_text()
    tables = {}
    for key, column, count in (("gen", 7, spec.generators), ("branch", 10, spec.lines)):
        block = re.search(r"mpc\." + key + r"\s*=\s*\[(.*?)\];", text, re.S)
        if block is None:
            raise ValueError("Missing nominal equipment table")
        rows = [
            [float(v) for v in line.split("%")[0].strip().rstrip(";").split()]
            for line in block[1].splitlines()
            if line.split("%")[0].strip()
        ]
        table = np.asarray(rows)
        values = table[:, column]
        if len(values) != count or not np.isin(values, [0, 1]).all():
            raise ValueError("Invalid nominal availability")
        tables[key] = table
    return tables


def nominal_status(spec):
    """Original availability, before the dataset samples additional outages."""
    tables = nominal_equipment(spec)
    return tables["gen"][:, 7].astype(int), tables["branch"][:, 10].astype(int)


def network_spec(case):
    dataset = case.get("dataset")
    from_dataset = None
    if dataset:
        matching = [name for name in SPECS if dataset == "gridfm/opf_small_" + name]
        if len(matching) != 1:
            raise ValueError("Unsupported OPF source dataset identity")
        from_dataset = matching[0]
    explicit = case.get("network")
    if explicit and from_dataset and explicit != from_dataset:
        raise ValueError("Case and dataset network identities disagree")
    name = explicit or from_dataset or "case14_ieee"
    if name not in SPECS:
        raise ValueError("Physical adapter is not verified for this network")
    return SPECS[name]
