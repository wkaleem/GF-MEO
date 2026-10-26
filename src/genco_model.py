"""GENCO model loading, network inputs and inference."""

from __future__ import annotations

from pathlib import Path
import copy
import itertools
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "vendor/gridfm-gridfm-graphkit-4e7fbcf"))

import numpy as np
import pandas as pd
import torch
import yaml

from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.models.utils import ComputeBranchFlow
from gridfm_graphkit.datasets.hetero_preprocess import (
    build_hetero_data_for_scenario,
    merge_agg_gen_into_bus,
)
from gridfm_graphkit.datasets.normalizers import HeteroDataMVANormalizer
from gridfm_graphkit.datasets.task_transforms import OptimalPowerFlowTransforms
from gridfm_graphkit.datasets.globals import VM_OUT, VA_OUT, QG_OUT, RATE_A

try:
    from .networks import network_spec, nominal_status, nominal_equipment
    from .reactive_generation import allocation_coefficients
except ImportError:
    from networks import network_spec, nominal_status, nominal_equipment
    from reactive_generation import allocation_coefficients

try:
    from .ac_constraints import ac_quantities, check_solution
except ImportError:
    from ac_constraints import ac_quantities, check_solution


EDGE = ("bus", "connects", "bus")
COST_COLUMNS = ["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2"]


def checkpoint_files(network, model_size, training_seed, checkpoint_root=None):
    """Resolve one pinned release bundle; never substitute weights or scales."""
    networks = (
        "case14_ieee",
        "case30_ieee",
        "case57_ieee",
        "case118_ieee",
        "case500_goc",
        "case2000_goc",
    )
    if (
        network not in networks
        or model_size not in ("small", "base")
        or type(training_seed) is not int
        or training_seed not in (0, 1)
    ):
        raise ValueError("Unknown released OPF network, model size or training seed")
    inventory = json.loads(Path(__file__).with_name("genco_models.json").read_text())
    if (
        inventory["repository"] != "gridfm/genco-opf-datakit"
        or inventory["revision"] != "2d2401572f6a2d0c1be79841b2def99cf45e1639"
    ):
        raise ValueError("Unexpected OPF release identity")
    prefix = f"{network}/{model_size}/seed{training_seed}/"
    records = {
        Path(r["path"]).name: r
        for r in inventory["records"]
        if r["path"].startswith(prefix)
    }
    weight = (
        "best_model_state_dict.pt"
        if network == "case2000_goc" and training_seed == 0
        else "last.pt"
    )
    names = {
        "config": f"{network}_{model_size}_seed{training_seed}.yaml",
        "checkpoint": weight,
        "normalizer": "normalizer_stats.pt",
    }
    if set(records) != set(names.values()):
        raise ValueError("Incomplete or ambiguous pinned OPF bundle")
    if checkpoint_root is None and (network, model_size, training_seed) == (
        "case14_ieee",
        "small",
        0,
    ):
        folder = ROOT / "models/opf_case14_small_seed0"
    else:
        root = (
            ROOT / "models/opf_inventory"
            if checkpoint_root is None
            else Path(checkpoint_root)
        )
        folder = root / network / model_size / f"seed{training_seed}"
    return {
        kind: dict(records[name], local_path=str(folder / name))
        for kind, name in names.items()
    }


def load_opf_bundle(
    dtype=torch.float64,
    *,
    device="cpu",
    network="case14_ieee",
    model_size="small",
    training_seed=0,
    checkpoint_root=None,
):
    """Strictly load a pinned released OPF bundle, including its saved scales.

    A successful load validates artifacts, not support by the UC case adapter.
    The released GOC2000 seed0 bundles use their best-validation weights.
    """
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("OPF loading supports float32 or float64 arithmetic")
    if torch.device(device).type != "cpu":
        raise ValueError("Only CPU model execution is supported")
    files = checkpoint_files(network, model_size, training_seed, checkpoint_root)
    for record in files.values():
        path = Path(record["local_path"])
        if (
            path.stat().st_size != record["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise ValueError(
                "OPF artifact does not match the pinned release: " + str(path)
            )
    args = NestedNamespace(
        **yaml.safe_load(Path(files["config"]["local_path"]).read_text())
    )
    if args.task.task_name != "OptimalPowerFlow":
        raise ValueError("OPF checkpoint/config required")
    if args.data.networks != [network] or args.seed != training_seed:
        raise ValueError(
            "OPF configuration does not match the selected network and seed"
        )
    if args.data.normalization != "HeteroDataMVANormalizer":
        raise ValueError("Unsupported OPF normalization")
    if args.model.type != "GNS_heterogeneous" or args.model.hidden_size != (
        {"small": 24, "base": 48}[model_size]
    ):
        raise ValueError("OPF architecture does not match the selected model size")
    model = GNS_heterogeneous(args)
    state = torch.load(
        files["checkpoint"]["local_path"], map_location="cpu", weights_only=True
    )
    stripped = {k.removeprefix("model."): v for k, v in state.items()}
    if len(stripped) != len(state):
        raise ValueError("Ambiguous checkpoint parameter names after prefix removal")
    model.load_state_dict(stripped, strict=True)
    model = model.to(dtype=dtype).eval().requires_grad_(False)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("Nonfinite checkpoint weights")
    normalizer = HeteroDataMVANormalizer(args)
    stats = torch.load(
        files["normalizer"]["local_path"], map_location="cpu", weights_only=True
    )
    if set(stats) != {network}:
        raise ValueError("Saved normalizer has no entry for the selected network")
    normalizer.fit_from_dict(stats[network])
    if not all(
        np.isfinite(v) and v > 0
        for v in (normalizer.baseMVA, normalizer.baseMVA_orig, normalizer.vn_kv_max)
    ):
        raise ValueError("Saved normalization scales must be finite and positive")
    if normalizer.baseMVA_orig != args.data.baseMVA:
        raise ValueError("Configuration and saved normalizer power bases disagree")
    model.artifact_provenance = dict(
        network=network,
        model_size=model_size,
        training_seed=training_seed,
        release_revision="2d2401572f6a2d0c1be79841b2def99cf45e1639",
        files=files,
        parameter_count=sum(p.numel() for p in model.parameters()),
        dtype=str(dtype),
        device="cpu",
    )
    return args, model, normalizer


class FrozenOPF:
    """A released GENCO OPF network with commitment and cost inputs.

    case: raw bus/gen/branch records, baseMVA=100, optional T and T-by-bus pd/qd.
    Generator rows must retain original, contiguous zero-based idx order.
    cost_order=(1,0) assigns original cost row1 to eligible target0 and row0 to
    eligible target1, even when one of those generators is offline.
    forward returns flat numpy arrays pg/qg/vm/va/uc plus independent diagnostics.
    """

    def __init__(self, case, *, dtype=torch.float64, model_bundle=None):
        self.case = copy.deepcopy(case)
        self.dtype = dtype
        self.spec = network_spec(case)
        self.model_bundle = (
            load_opf_bundle(dtype, network=self.spec.name)
            if model_bundle is None
            else model_bundle
        )
        self.args, self.model, self.normalizer = self.model_bundle
        if self.args.data.networks != [self.spec.name]:
            raise ValueError(
                f"This case requires an {self.spec.display} checkpoint and saved normalizer"
            )
        if next(self.model.parameters()).dtype != dtype:
            raise ValueError("Shared model dtype differs from requested dtype")
        self.base = float(self.normalizer.baseMVA)
        self.bus = pd.DataFrame(case["bus"]).reset_index(drop=True)
        self.gen = pd.DataFrame(case["gen"]).reset_index(drop=True)
        self.branch = pd.DataFrame(case["branch"]).reset_index(drop=True)
        self.nb, self.ng, self.nl = len(self.bus), len(self.gen), len(self.branch)
        if (self.nb, self.ng, self.nl) != (
            self.spec.buses,
            self.spec.generators,
            self.spec.lines,
        ) or float(case.get("baseMVA", 100)) != 100:
            raise ValueError(
                f"This checkpoint requires the original 100-MVA {self.spec.display} equipment layout"
            )
        if not np.array_equal(self.bus.bus.to_numpy(), np.arange(self.nb)):
            raise ValueError("Bus IDs must be contiguous zero-based IDs")
        if not np.array_equal(self.gen.idx.to_numpy(), np.arange(self.ng)):
            raise ValueError("Generator IDs must be contiguous zero-based IDs")
        if not np.array_equal(self.branch.idx.to_numpy(), np.arange(self.nl)):
            raise ValueError("Branch IDs must be contiguous zero-based IDs")
        scenario = int(case.get("scenario", 0))
        for frame in (self.bus, self.gen, self.branch):
            if "scenario" not in frame:
                frame["scenario"] = scenario
            elif not (frame.scenario == scenario).all():
                raise ValueError("Record scenario IDs disagree with case identity")
        self.nominal_gen_status, self.nominal_branch_status = nominal_status(self.spec)
        equipment = nominal_equipment(self.spec)
        if not np.array_equal(self.gen.bus.to_numpy(), equipment["gen"][:, 0] - 1):
            raise ValueError(
                "Generator bus assignments differ from the nominal network"
            )
        if not np.array_equal(
            self.branch[["from_bus", "to_bus"]].to_numpy(),
            equipment["branch"][:, :2] - 1,
        ):
            raise ValueError("Line endpoints/order differ from the nominal network")
        self.ref_buses = np.flatnonzero(self.bus.REF.to_numpy() == 1)
        if len(self.ref_buses) != 1:
            raise ValueError("Exactly one reference bus required")
        self.ref_bus = int(self.ref_buses[0])
        self.ref_gens = np.flatnonzero(
            (self.gen.bus.to_numpy() == self.ref_bus) & (self.nominal_gen_status == 1)
        )
        if not len(self.ref_gens):
            raise ValueError("At least one available reference generator required")
        self.actual_cost_rows = self.gen[COST_COLUMNS].to_numpy(dtype=float).copy()
        self.eligible = np.flatnonzero(
            np.any(self.actual_cost_rows[:, 1:] != 0, axis=1)
        )
        self.T = int(case.get("T", 1))
        self.pd = np.asarray(case.get("pd", [self.bus.Pd.to_numpy()] * self.T), float)
        self.qd = np.asarray(case.get("qd", [self.bus.Qd.to_numpy()] * self.T), float)
        if self.pd.shape != (self.T, self.nb) or self.qd.shape != self.pd.shape:
            raise ValueError("Demand must have shape T by number of buses")
        if not np.isfinite(self.pd).all() or not np.isfinite(self.qd).all():
            raise ValueError("Nonfinite demand")
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

    def cost_orders(self, commitment=None):
        """Enumerate distinct visible cost assignments, retaining a representative."""
        if len(self.eligible) > 8:
            raise ValueError(
                "Exhaustive cost-permutation enumeration is disabled above eight eligible generators; supply the desired cost order explicitly"
            )
        u = self._status(commitment)
        online_eligible = self.eligible[u[self.eligible] == 1]
        seen, orders = set(), []
        for order in itertools.permutations(self.eligible.tolist()):
            assigned = dict(zip(self.eligible.tolist(), order))
            key = tuple(
                self.actual_cost_rows[assigned[g]].tobytes() for g in online_eligible
            )
            if key not in seen:
                seen.add(key)
                orders.append(order)
        return orders

    def _status(self, commitment):
        u = (
            self.gen.in_service.to_numpy(float).copy()
            if commitment is None
            else np.asarray(commitment, float)
        )
        if u.shape != (self.ng,) or not np.isin(u, [0.0, 1.0]).all():
            raise ValueError(
                "Commitment must be a binary vector in original generator order"
            )
        if not np.all(u[self.ref_gens] == 1):
            raise ValueError("Reference generator must remain online")
        if np.any(u > self.nominal_gen_status):
            raise ValueError("A nominally unavailable generator cannot be activated")
        return u

    def prepare(
        self,
        t=0,
        commitment=None,
        cost_order=None,
        *,
        require_supported_outage_family=True,
    ):
        if (
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, np.integer))
            or not 0 <= t < self.T
        ):
            raise IndexError("Period outside horizon")
        u = self._status(commitment)
        order = (
            tuple(self.eligible.tolist()) if cost_order is None else tuple(cost_order)
        )
        if sorted(order) != self.eligible.tolist():
            raise ValueError(
                "cost_order must permute the fixed original eligible generator indices"
            )
        b, g, e = self.bus.copy(), self.gen.copy(), self.branch.copy()
        branch_status = e.br_status.to_numpy(float)
        if not np.isin(branch_status, [0.0, 1.0]).all():
            raise ValueError("Branch status must be binary")
        if np.any(branch_status > self.nominal_branch_status):
            raise ValueError("A nominally unavailable line cannot be activated")
        outage_count = int(
            np.sum(self.nominal_gen_status - u)
            + np.sum(self.nominal_branch_status - branch_status)
        )
        if (
            require_supported_outage_family
            and not 1 <= outage_count <= self.spec.maximum_outages
        ):
            raise ValueError(
                f"{self.spec.display} release samples one to {self.spec.maximum_outages} component outages"
            )
        # Connectivity is necessary; this is not a proof of AC feasibility.
        neighbors = {i: set() for i in range(self.nb)}
        for row in e.loc[branch_status == 1].itertuples():
            neighbors[int(row.from_bus)].add(int(row.to_bus))
            neighbors[int(row.to_bus)].add(int(row.from_bus))
        reached, pending = {self.ref_bus}, [self.ref_bus]
        while pending:
            for neighbor in neighbors[pending.pop()] - reached:
                reached.add(neighbor)
                pending.append(neighbor)
        if len(reached) != self.nb:
            raise ValueError("Disconnected graph is outside the release sampler")
        b["Pd"], b["Qd"] = self.pd[t], self.qd[t]
        g["in_service"] = u
        g.loc[self.eligible, COST_COLUMNS] = self.actual_cost_rows[list(order)]
        online_buses = set(g.loc[u == 1, "bus"].astype(int))
        b["PV"] = [
            float(int(i) in online_buses and int(i) != self.ref_bus) for i in b.bus
        ]
        b["PQ"] = 1.0 - b.PV - b.REF
        # Delete solved targets before construction; REF angle and zero PQ Qg
        # remain genuine known inputs, not target features.
        b["Qg"], b["Vm"] = 0.0, 0.0
        b.loc[b.REF == 0, "Va"] = 0.0
        g["p_mw"], g["q_mvar"] = 0.0, 0.0
        for col in ("pf", "qf", "pt", "qt"):
            e[col] = 0.0
        # Source merge sums all original Q-bound rows, including offline units.
        # Preserve that trained feature convention; physical Q bounds use u.
        b = b.drop(columns=["min_q_mvar", "max_q_mvar"], errors="ignore")
        merged = merge_agg_gen_into_bus(b, g).sort_values("bus")
        scenario = int(self.case.get("scenario", 0))
        data = build_hetero_data_for_scenario(scenario, merged, g, e)
        self.normalizer.transform(data)
        data = OptimalPowerFlowTransforms(self.args)(data)
        for kind in ("bus", "gen"):
            if not torch.all(
                data[kind].x[data.mask_dict[kind]] == self.args.data.mask_value
            ):
                raise AssertionError("Target feature survived masking")
        if not torch.all(
            data[EDGE].edge_attr[data.mask_dict["branch"]] == self.args.data.mask_value
        ):
            raise AssertionError("Branch target survived masking")
        for store in data.stores:
            if "y" in store:
                del store["y"]
            for key, value in list(store.items()):
                if isinstance(value, torch.Tensor) and value.is_floating_point():
                    store[key] = value.to(dtype=self.dtype)
        active_gens = np.flatnonzero(u)
        active_branches = np.flatnonzero(branch_status)
        directed_branches = np.concatenate([active_branches, active_branches])
        directed_from = np.concatenate(
            [np.ones(len(active_branches), bool), np.zeros(len(active_branches), bool)]
        )
        rates = e.rate_a.to_numpy(float)[directed_branches].copy()
        expected_edges = np.concatenate(
            [
                e.loc[branch_status == 1, ["from_bus", "to_bus"]].to_numpy().T,
                e.loc[branch_status == 1, ["to_bus", "from_bus"]].to_numpy().T,
            ],
            axis=1,
        )
        if not np.array_equal(data[EDGE].edge_index.numpy(), expected_edges):
            raise AssertionError("Unexpected directed branch order")
        q_weights, q_offset = allocation_coefficients(
            self.gen.bus.to_numpy(int),
            u,
            self.gen.min_q_mvar.to_numpy(float),
            self.gen.max_q_mvar.to_numpy(float),
        )
        return {
            "data": data,
            "owner": id(self),
            "u": u.copy(),
            "active_generators": active_gens,
            "q_allocation_weights": q_weights,
            "q_allocation_offset": q_offset,
            "active_branches": active_branches,
            "directed_branches": directed_branches,
            "directed_from": directed_from,
            "directed_rate_mva": rates,
            "virtual_cost_rows": g[COST_COLUMNS].to_numpy(float),
            "cost_order": order,
            "outage_count": outage_count,
            "t": int(t),
        }

    def static_physical_case(self, t, commitment):
        """Native one-period case for independent AC checking, with no UC claim."""
        u = self._status(commitment)
        return {
            "T": 1,
            "baseMVA": 100.0,
            "bus": self.bus.to_dict("records"),
            "gen": self.gen.to_dict("records"),
            "branch": self.branch.to_dict("records"),
            "pd": [self.pd[t].tolist()],
            "qd": [self.qd[t].tolist()],
            "initial_on": u.tolist(),
            "fixed_on": [],
            "ramp_ratio": 1.0,
            "min_up": [1] * self.ng,
            "min_down": [1] * self.ng,
            "initial_forced_on": [0] * self.ng,
            "initial_forced_off": [0] * self.ng,
            "startup_cost": [0.0] * self.ng,
            "shutdown_cost": [0.0] * self.ng,
            "initial_pg_ramp_enforced": False,
        }

    def forward(self, t=0, commitment=None, cost_order=None, *, prepared=None):
        item = self.prepare(t, commitment, cost_order) if prepared is None else prepared
        if item.get("owner") != id(self):
            raise ValueError("Prepared graph belongs to a different adapter")
        data, u = item["data"], item["u"]
        with torch.no_grad():
            prediction = self.model(
                data.x_dict, data.edge_index_dict, data.edge_attr_dict, data.mask_dict
            )
            if not all(torch.isfinite(value).all() for value in prediction.values()):
                raise FloatingPointError("Nonfinite OPF prediction")
            bp, bq = ComputeBranchFlow()(
                prediction["bus"], data.edge_index_dict[EDGE], data.edge_attr_dict[EDGE]
            )
        bus_output = prediction["bus"].detach().numpy()
        pg = np.zeros(self.ng)
        pg[item["active_generators"]] = (
            prediction["gen"][:, 0].detach().numpy() * self.base
        )
        qg = (
            bus_output[self.gen.bus.to_numpy(int), QG_OUT]
            * self.base
            * item["q_allocation_weights"]
            + item["q_allocation_offset"]
        )
        state = {
            "pg": pg,
            "qg": qg,
            "vm": bus_output[:, VM_OUT].copy(),
            "va": bus_output[:, VA_OUT].copy(),
            "uc": u.copy(),
        }
        physical_case = self.static_physical_case(item["t"], u)
        checked = check_solution(
            physical_case, {key: value[None, :] for key, value in state.items()}
        )
        raw = ac_quantities(physical_case, state["vm"][None, :], state["va"][None, :])
        model_sf, model_st = np.zeros(self.nl, complex), np.zeros(self.nl, complex)
        directed_power = (bp.detach().numpy() + 1j * bq.detach().numpy()) * self.base
        model_sf[item["directed_branches"][item["directed_from"]]] = directed_power[
            item["directed_from"]
        ]
        model_st[item["directed_branches"][~item["directed_from"]]] = directed_power[
            ~item["directed_from"]
        ]
        virtual = item["virtual_cost_rows"]
        virtual_cost = float(
            np.sum(virtual[:, 0] * u + virtual[:, 1] * pg + virtual[:, 2] * pg**2)
        )
        state.update(
            {
                "actual_cost": checked["total_cost_author_constant"],
                "actual_cost_commitment_constant": checked[
                    "total_cost_commitment_constant"
                ],
                "virtual_cost": virtual_cost,
                "native_check": checked,
                "branch_pf_mw": model_sf.real,
                "branch_qf_mvar": model_sf.imag,
                "branch_pt_mw": model_st.real,
                "branch_qt_mvar": model_st.imag,
                "directed_rate_mva": item["directed_rate_mva"].copy(),
                "raw_physics_flow_max_error_mva": float(
                    max(
                        np.max(np.abs(model_sf - raw["from_mva"][0])),
                        np.max(np.abs(model_st - raw["to_mva"][0])),
                    )
                ),
                "metadata": {
                    "scenario": self.case.get("scenario"),
                    "load_scenario_idx": self.case.get("load_scenario_idx"),
                    "split": self.case.get("split"),
                    "period": item["t"],
                    "eligible_generator_indices": self.eligible.tolist(),
                    "cost_order": list(item["cost_order"]),
                    "outage_count": item["outage_count"],
                    "active_generator_indices": item["active_generators"].tolist(),
                    "all_targets_removed": all(
                        "y" not in store for store in data.stores
                    ),
                    "normalizer_baseMVA": self.base,
                    "dtype": str(self.dtype),
                    "weights_frozen": all(
                        not p.requires_grad for p in self.model.parameters()
                    ),
                    "actual_cost_convention": "Unconditional original c0; generation coefficients preserve the source order",
                    "reactive_generation": "Online capacity-width affine allocation of bus totals; equal-share mismatch when total width is zero; no clipping",
                    "official_q_limit_aggregation_includes_offline_rows": True,
                    "model_rating_vs_raw_max_error_mva": float(
                        np.max(
                            np.abs(
                                data[EDGE].edge_attr[:, RATE_A].numpy() * self.base
                                - item["directed_rate_mva"]
                            )
                        )
                    ),
                },
            }
        )
        return state
