"""Auxiliary input variables for the released GENCO OPF models.

The actual case is immutable. Virtual demand, impedance and cost coefficients
are neural inputs only. Use TorchAC/physical_check with the original case to
assess physical feasibility. Interior cost mixtures are outside the released
discrete cost-permutation sampler, even if load/impedance boxes are respected.

Variable features are recomputed in float64 before the trained float64 model.
This is a disclosed precision change from official float32 preprocessing, not
a change to the checkpoint weights. Zero offsets reproduce the same raw inputs
but may differ slightly from the original rounded features and predictions.
"""

from __future__ import annotations
import copy
import hashlib
import re
from pathlib import Path
import numpy as np
import torch
from torch_geometric.data import Batch

from opf_adapter import FrozenOPF, EDGE, COST_COLUMNS, ROOT
from input_domain import virtual_load_box
from gridfm_graphkit.datasets.globals import (
    PD_H,
    QD_H,
    C0_H,
    C2_H,
    YFF_TT_R,
    YFT_TF_I,
    VM_OUT,
    VA_OUT,
    QG_OUT,
)


def signed_log_cost(coefficients):
    """Released signed-log values with the analytic first derivative at zero.

    sign(c)*log1p(abs(c)) is differentiable at zero with derivative one,
    but the literal composition's autograd derivative is zero there.
    The zero branch preserves every forward value and the correct derivative.
    """
    transformed = torch.sign(coefficients) * torch.log1p(torch.abs(coefficients))
    return torch.where(coefficients == 0, coefficients, transformed)


def _nominal_tables(spec=None):
    path = ROOT / (
        "sources/distribution_audit/case14_nominal_repro.m"
        if spec is None
        else spec.nominal_path
    )
    text = path.read_text()
    tables = {}
    for name in ("bus", "branch"):
        match = re.search(r"mpc\." + name + r"\s*=\s*\[(.*?)\];", text, re.S)
        if match is None:
            raise ValueError(f"Missing nominal {name} table")
        tables[name] = np.asarray(
            [
                [
                    float(value)
                    for value in line.split("%")[0].strip().rstrip(";").split()
                ]
                for line in match.group(1).splitlines()
                if line.split("%")[0].strip()
            ]
        )
    return tables, {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def directed_admittances(r, x, charging, tap, shift_degrees):
    """Consistent pi-model Y columns, all forward edges then all reverse edges."""
    ys = torch.reciprocal(torch.complex(r, x))
    ratio = torch.where(tap == 0, torch.ones_like(tap), tap)
    a = torch.polar(ratio, torch.deg2rad(shift_degrees))
    ytt = ys + torch.complex(torch.zeros_like(charging), charging / 2)
    yff = ytt / ratio.square()
    yft, ytf = -ys / torch.conj(a), -ys / a
    own, mutual = torch.cat([yff, ytt]), torch.cat([yft, ytf])
    return torch.stack([own.real, own.imag, mutual.real, mutual.imag], dim=1)


class FrozenVirtualOPF:
    """Packed coordinates x=[load fractional offsets, r/x offsets, cost lambda].

    Load coordinates list all nonzero Pd then all nonzero Qd, including generator
    buses. Impedance coordinates list active nonzero r then active nonzero x.
    Zero-valued fields and offline branches remain unchanged. Lambda=0 uses the
    original cost rows; lambda=1 reverses the eligible row order. Intermediate
    lambda mixes RAW coefficients before signed-log normalization.

    bounds(t) defines local fractional boxes. Impedance boxes intersect the
    nominal +/-20% generator range. Load boxes use one admissible common global
    scale per period, preserving the documented [.8,1.2] local-factor box.
    This is feature-domain containment, not joint distribution/feasibility proof.
    Callers must impose these bounds; forward itself does not clamp parameters.
    """

    def __init__(
        self,
        case,
        *,
        dtype=torch.float64,
        model_bundle=None,
        load_radius=0.01,
        rx_radius=0.01,
    ):
        if dtype != torch.float64:
            raise ValueError("The smooth virtual adapter currently requires float64")
        if not (0 <= load_radius < 1 and 0 <= rx_radius < 1):
            raise ValueError("Local fractional radii must lie in [0,1)")
        self.case = copy.deepcopy(case)
        self.dtype = dtype
        self.adapter = FrozenOPF(case, dtype=dtype, model_bundle=model_bundle)
        self.model = self.adapter.model
        self.nb, self.ng = self.adapter.nb, self.adapter.ng
        self.T, self.base = self.adapter.T, self.adapter.base
        self.load_radius, self.rx_radius = float(load_radius), float(rx_radius)
        self.nominal, self.nominal_provenance = _nominal_tables(self.adapter.spec)
        if len(self.nominal["branch"]) != self.adapter.nl:
            raise ValueError("Nominal branch order does not match the physical case")
        pairs = self.adapter.branch[["from_bus", "to_bus"]].to_numpy(int)
        if not np.array_equal(pairs, self.nominal["branch"][:, :2].astype(int) - 1):
            raise ValueError("Nominal and actual branch IDs/order differ")
        self.load_mask = np.concatenate(
            [self.adapter.pd[0] != 0, self.adapter.qd[0] != 0]
        )
        nominal_load_mask = np.concatenate(
            [self.nominal["bus"][:, 2] != 0, self.nominal["bus"][:, 3] != 0]
        )
        if not np.array_equal(self.load_mask, nominal_load_mask):
            raise ValueError(
                "Actual/nominal load zero patterns differ; positive scaling cannot change zero support"
            )
        for t in range(self.T):
            if not np.array_equal(
                self.load_mask,
                np.concatenate([self.adapter.pd[t] != 0, self.adapter.qd[t] != 0]),
            ):
                raise ValueError(
                    "Nonzero-load support must remain fixed over the horizon"
                )
        self.load_indices = torch.tensor(
            np.flatnonzero(self.load_mask), dtype=torch.long
        )
        self.load_coordinates = [
            {"field": "Pd" if j < self.nb else "Qd", "bus": int(j % self.nb)}
            for j in self.load_indices.tolist()
        ]
        self.active = np.flatnonzero(self.adapter.branch.br_status.to_numpy() == 1)
        self.active_index = torch.tensor(self.active, dtype=torch.long)
        raw_rx = self.adapter.branch[["r", "x"]].to_numpy(float)
        nominal_rx = self.nominal["branch"][:, 2:4]
        if np.any((nominal_rx == 0) != (raw_rx == 0)):
            raise ValueError("Actual/nominal impedance zero patterns differ")
        if np.any(raw_rx < 0) or np.any(nominal_rx < 0):
            raise NotImplementedError(
                "This adapter supports native nonnegative impedances"
            )
        self.rx_coordinates = [
            {"field": field, "branch": int(j)}
            for col, field in enumerate(("r", "x"))
            for j in self.active
            if raw_rx[j, col] != 0
        ]
        self.rx_flat_indices = torch.tensor(
            [
                entry["branch"] + (0 if entry["field"] == "r" else self.adapter.nl)
                for entry in self.rx_coordinates
            ],
            dtype=torch.long,
        )
        self.rx_raw = self.tensor(np.concatenate([raw_rx[:, 0], raw_rx[:, 1]]))
        self.rx_nominal = self.tensor(
            np.concatenate([nominal_rx[:, 0], nominal_rx[:, 1]])
        )
        chosen = self.rx_flat_indices
        ratio = self.rx_raw[chosen] / self.rx_nominal[chosen]
        if bool(((ratio < 0.8 - 1e-12) | (ratio > 1.2 + 1e-12)).any()):
            raise ValueError(
                "Original active r/x lie outside the logged nominal +/-20% range"
            )
        self.charging, self.tap, self.shift = [
            self.tensor(self.adapter.branch[col].to_numpy(float)[self.active])
            for col in ("b", "tap", "shift")
        ]
        if bool((self.tap < 0).any()):
            raise ValueError("Negative transformer tap")
        self.ngen_bus = torch.tensor(
            self.adapter.gen.bus.to_numpy(int), dtype=torch.long
        )
        self.cost_original = self.tensor(self.adapter.actual_cost_rows)
        self.cost_swapped = self.cost_original.clone()
        eligible = self.adapter.eligible
        self.cost_swapped[torch.tensor(eligible)] = self.cost_original[
            torch.tensor(eligible[::-1].copy())
        ]
        self.n_load, self.n_rx = len(self.load_coordinates), len(self.rx_coordinates)
        self.n_parameters = self.n_load + self.n_rx + 1
        self.cost_index = self.n_parameters - 1
        self._cache = {}
        self._batch_cache = {}
        # Choose a nonempty virtual-input box for each period. Physical demands
        # remain unchanged and need not themselves be admissible model inputs.
        self._bounds = [self._make_bounds(t) for t in range(self.T)]
        self.metadata = {
            "parameter_layout": {
                "load": self.load_coordinates,
                "rx": self.rx_coordinates,
                "cost_mix_index": self.cost_index,
            },
            "n_parameters": self.n_parameters,
            "nominal_source": self.nominal_provenance,
            "load_box_construction": [entry[2] for entry in self._bounds],
            "preprocessing": "Variable loads, pi admittances and signed-log costs recomputed in float64; other features retain official rounded values",
            "physical_case_modified": False,
            "weights_frozen": True,
            "cost_mixture_support": "Only lambda=0 or 1 is a released discrete permutation; interior mixtures are out of distribution",
            "feature_support_claim": "Local boxes inside documented marginal generator ranges; no joint-training-support or feasibility claim",
            "all_original_balance_equations_required": True,
            "forward_enforces_parameter_bounds": False,
        }

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype)

    def zero_parameters(self, cost_mix=0.0):
        """Literal zero offsets; not necessarily an admissible initial point."""
        result = torch.zeros(self.n_parameters, dtype=self.dtype)
        result[-1] = float(cost_mix)
        return result

    def split_parameters(self, x):
        if (
            x.shape != (self.n_parameters,)
            or x.dtype != self.dtype
            or x.device.type != "cpu"
        ):
            raise ValueError(
                f"Expected a float64 CPU vector of shape ({self.n_parameters},)"
            )
        return x[: self.n_load], x[self.n_load : -1], x[-1]

    def _make_bounds(self, t):
        actual = np.concatenate([self.adapter.pd[t], self.adapter.qd[t]])[
            self.load_mask
        ]
        nominal = np.concatenate(
            [self.nominal["bus"][:, 2], self.nominal["bus"][:, 3]]
        )[self.load_mask]
        ratios = actual / nominal
        load_lo, load_hi, metadata = virtual_load_box(
            ratios,
            self.load_radius,
            global_range=self.adapter.spec.global_load_range,
            local_range=(0.8, 1.2),
        )
        base = self.rx_raw[self.rx_flat_indices]
        nominal_rx = self.rx_nominal[self.rx_flat_indices]
        rx_lo = torch.maximum(
            torch.full_like(base, -self.rx_radius), 0.8 * nominal_rx / base - 1
        )
        rx_hi = torch.minimum(
            torch.full_like(base, self.rx_radius), 1.2 * nominal_rx / base - 1
        )
        lower = torch.cat([self.tensor(load_lo), rx_lo, self.tensor([0.0])])
        upper = torch.cat([self.tensor(load_hi), rx_hi, self.tensor([1.0])])
        if bool((lower > upper).any()):
            raise ValueError("Empty virtual-input coordinate interval")
        return lower, upper, metadata

    def bounds(self, t):
        lower, upper, _ = self._bounds[t]
        return lower.clone(), upper.clone()

    def prepare(self, t=0, commitment=None):
        if (
            isinstance(t, (bool, np.bool_))
            or not isinstance(t, (int, np.integer))
            or not 0 <= t < self.T
        ):
            raise IndexError("Period outside horizon")
        u = self.adapter._status(commitment)
        key = (int(t), tuple(u.tolist()))
        if key not in self._cache:
            item = self.adapter.prepare(t, u)
            item["active_generator_tensor"] = torch.tensor(
                item["active_generators"], dtype=torch.long
            )
            item["uc_tensor"] = self.tensor(u)
            item["q_allocation_weights_tensor"] = self.tensor(
                item["q_allocation_weights"]
            )
            item["q_allocation_offset_tensor"] = self.tensor(
                item["q_allocation_offset"]
            )
            self._cache[key] = item
        return self._cache[key]

    def virtual_features(self, t, commitment, x, *, prepared=None):
        """Return new feature tensors and virtual raw fields without inference."""
        z_load, z_rx, cost_mix = self.split_parameters(x)
        item = self.prepare(t, commitment) if prepared is None else prepared
        if (
            item.get("owner") != id(self.adapter)
            or item["t"] != t
            or not np.array_equal(item["u"], self.adapter._status(commitment))
        ):
            raise ValueError(
                "Prepared graph does not match requested period/commitment"
            )
        raw_load = self.tensor(np.concatenate([self.adapter.pd[t], self.adapter.qd[t]]))
        load = raw_load.index_copy(
            0, self.load_indices, raw_load[self.load_indices] * (1 + z_load)
        )
        rx = self.rx_raw.index_copy(
            0, self.rx_flat_indices, self.rx_raw[self.rx_flat_indices] * (1 + z_rx)
        )
        r, x_branch = rx[: self.adapter.nl], rx[self.adapter.nl :]
        y = directed_admittances(
            r[self.active_index],
            x_branch[self.active_index],
            self.charging,
            self.tap,
            self.shift,
        )
        normalized_y = y * (float(self.adapter.normalizer.baseMVA_orig) / self.base)
        costs = self.cost_original + cost_mix * (self.cost_swapped - self.cost_original)
        normalized_costs = signed_log_cost(costs)
        data = item["data"]
        bus, gen, edge = (
            data["bus"].x.clone(),
            data["gen"].x.clone(),
            data[EDGE].edge_attr.clone(),
        )
        bus[:, PD_H] = load[: self.nb] / self.base
        bus[:, QD_H] = load[self.nb :] / self.base
        gen[:, C0_H : C2_H + 1] = normalized_costs[item["active_generator_tensor"]]
        edge[:, YFF_TT_R : YFT_TF_I + 1] = normalized_y
        x_dict = dict(data.x_dict)
        x_dict.update(bus=bus, gen=gen)
        edge_attr = dict(data.edge_attr_dict)
        edge_attr[EDGE] = edge
        return {
            "x_dict": x_dict,
            "edge_attr_dict": edge_attr,
            "item": item,
            "virtual_pd": load[: self.nb],
            "virtual_qd": load[self.nb :],
            "virtual_r": r,
            "virtual_x": x_branch,
            "virtual_costs": costs,
            "virtual_directed_admittances": y,
            "cost_mix": cost_mix,
        }

    def forward(self, t, commitment, x, *, prepared=None):
        inputs = self.virtual_features(t, commitment, x, prepared=prepared)
        item = inputs["item"]
        data = item["data"]
        prediction = self.model(
            inputs["x_dict"],
            data.edge_index_dict,
            inputs["edge_attr_dict"],
            data.mask_dict,
        )
        return self._decode_state(prediction["bus"], prediction["gen"], inputs)

    def _decode_state(self, bus, gen, inputs):
        item = inputs["item"]
        pg = torch.zeros(self.ng, dtype=self.dtype).index_copy(
            0, item["active_generator_tensor"], gen[:, 0] * self.base
        )
        u = item["uc_tensor"]
        return {
            "pg": pg,
            "qg": bus[self.ngen_bus, QG_OUT]
            * self.base
            * item["q_allocation_weights_tensor"]
            + item["q_allocation_offset_tensor"],
            "vm": bus[:, VM_OUT],
            "va": bus[:, VA_OUT],
            "uc": u,
            "virtual_inputs": {
                key: inputs[key]
                for key in (
                    "virtual_pd",
                    "virtual_qd",
                    "virtual_r",
                    "virtual_x",
                    "virtual_costs",
                    "cost_mix",
                )
            },
        }

    def prepare_batch(self, requests):
        """Cache a disjoint PyG union for a fixed list of (period, commitment)."""
        if not requests:
            raise ValueError("At least one period/commitment request is required")
        for t, u in requests:
            self.prepare(t, u)
        key = tuple(
            (int(t), tuple(self.adapter._status(u).tolist())) for t, u in requests
        )
        if key not in self._batch_cache:
            items = [self.prepare(t, u) for t, u in key]
            graphs = [item["data"] for item in items]
            batch = Batch.from_data_list(graphs)
            # Mask collation must preserve a separate REF and status mask in
            # every graph. No edge may couple two different scenarios.
            for kind in ("bus", "gen", "branch", "PQ", "PV", "REF"):
                expected = torch.cat([data.mask_dict[kind] for data in graphs], dim=0)
                if not torch.equal(batch.mask_dict[kind], expected):
                    raise ValueError(f"Unexpected batch mask collation: {kind}")
            for (source, _, target), edge_index in batch.edge_index_dict.items():
                if not torch.equal(
                    batch[source].batch[edge_index[0]],
                    batch[target].batch[edge_index[1]],
                ):
                    raise ValueError("Graph collation introduced cross-scenario edges")
            self._batch_cache[key] = {
                "requests": key,
                "items": items,
                "batch": batch,
                "owner": id(self),
                "bus_sizes": [data["bus"].num_nodes for data in graphs],
                "gen_sizes": [data["gen"].num_nodes for data in graphs],
            }
        return self._batch_cache[key]

    def forward_batch(self, requests, parameters, *, prepared=None):
        """One frozen forward for independent dynamic virtual-input graphs.

        parameters is an N-by-n_parameters float64 CPU tensor. The returned
        list has one physical-unit state dict per request, in the same order.
        Both node features AND directed admittance features vary independently
        per graph; the cached union contains only fixed topology/masks/features.
        """
        item = self.prepare_batch(requests) if prepared is None else prepared
        key = tuple(
            (int(t), tuple(self.adapter._status(u).tolist())) for t, u in requests
        )
        if item["owner"] != id(self) or item["requests"] != key:
            raise ValueError(
                "Prepared virtual batch does not match this adapter/request list"
            )
        if parameters.shape != (len(key), self.n_parameters):
            raise ValueError(
                f"Expected parameter matrix of shape ({len(key)},{self.n_parameters})"
            )
        features = [
            self.virtual_features(t, u, parameters[j], prepared=item["items"][j])
            for j, (t, u) in enumerate(key)
        ]
        x_dict = {
            kind: torch.cat([entry["x_dict"][kind] for entry in features], dim=0)
            for kind in ("bus", "gen")
        }
        batch = item["batch"]
        edge_attr = dict(batch.edge_attr_dict)
        edge_attr[EDGE] = torch.cat(
            [entry["edge_attr_dict"][EDGE] for entry in features], dim=0
        )
        prediction = self.model(
            x_dict, batch.edge_index_dict, edge_attr, batch.mask_dict
        )
        buses = torch.split(prediction["bus"], item["bus_sizes"], dim=0)
        generators = torch.split(prediction["gen"], item["gen_sizes"], dim=0)
        return [
            self._decode_state(bus, gen, inputs)
            for bus, gen, inputs in zip(buses, generators, features)
        ]
