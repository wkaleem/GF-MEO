"""GENCO inputs, inference derivatives and independent period checks.

The corrected period solver composes this predictor with the AC projection.
Input coordinates change neural conditioning, not the original network data.
"""

from __future__ import annotations
import copy
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from opf_virtual_adapter import FrozenVirtualOPF, signed_log_cost
from opf_adapter import EDGE
from gridfm_graphkit.datasets.globals import C0_H, C2_H
from torch_ac import TorchAC
from physical_check import check_solution, ac_quantities

ROOT = Path(__file__).resolve().parents[1]


def serial(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def atomic_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(record, default=serial, indent=2, allow_nan=False) + "\n"
    )
    temp.replace(path)


class PeriodOracle:
    def __init__(
        self,
        case,
        t,
        u,
        radius,
        convention,
        *,
        optimize_linear_costs=False,
        model_bundle=None,
        jacobian_mode="state_batched",
    ):
        if convention not in ("literal_period", "standard_physical"):
            raise ValueError("Unknown period convention")
        if jacobian_mode not in ("state_batched", "rowwise"):
            raise ValueError("Unknown Jacobian mode")
        self.jacobian_mode = jacobian_mode
        self.case = copy.deepcopy(case)
        self.t = t
        self.u = np.asarray(u, float)
        self.convention = convention
        self.adapter = FrozenVirtualOPF(
            case, load_radius=radius, rx_radius=radius, model_bundle=model_bundle
        )
        self.physics = TorchAC(case)
        if convention == "literal_period":
            if bool((self.physics.pmin < 0).any()):
                raise ValueError(
                    "Literal-period pilot requires nonnegative original Pmin"
                )
            if not bool(
                torch.isfinite(self.physics.rate).all()
                and (self.physics.rate > 0).all()
            ):
                raise ValueError(
                    "Literal-period pilot requires finite positive active line limits"
                )
        self.prepared = self.adapter.prepare(t, u)
        lo, hi = self.adapter.bounds(t)
        # The legacy cost interpolation is FIXED at zero, never optimized.
        lo[-1] = hi[-1] = 0.0
        self.legacy_dimension = len(lo)
        self.cost_generators = torch.as_tensor(
            self.adapter.adapter.eligible, dtype=torch.long
        )
        self.optimize_linear_costs = optimize_linear_costs
        initial = torch.zeros_like(lo)
        if optimize_linear_costs and len(self.cost_generators):
            costs = self.adapter.cost_original[self.cost_generators, 1]
            lower = torch.full_like(costs, float(costs.min()))
            upper = torch.full_like(costs, float(costs.max()))
            offline = torch.as_tensor(self.u[self.cost_generators.numpy()] == 0)
            lower[offline] = costs[offline]
            upper[offline] = costs[offline]
            lo = torch.cat((lo, lower))
            hi = torch.cat((hi, upper))
            initial = torch.cat((initial, costs))
        self.lo, self.hi = lo, hi
        # A valid virtual-input box may exclude the original input values.
        # Project raw coordinates before scaling or eliminating fixed entries.
        initial = torch.minimum(torch.maximum(initial, lo), hi)
        self.initial_offsets = initial.clone()
        self.mid = (lo + hi) / 2
        self.half = (hi - lo) / 2
        self.free = torch.nonzero(self.half > 1e-14).flatten()
        self.z0 = (
            (initial[self.free] - self.mid[self.free]) / self.half[self.free]
        ).numpy()
        self.cache = None
        self.calls = 0

    def domain_record(self):
        return dict(
            period=self.t,
            pattern=self.u,
            load_box=self.adapter.metadata["load_box_construction"][self.t],
            lower=self.lo,
            upper=self.hi,
            initial_offsets=self.initial_offsets,
            original_physical_data_unchanged=True,
        )

    def forward(self, z):
        x = self.mid.index_copy(
            0, self.free, self.mid[self.free] + self.half[self.free] * z
        )
        if self.optimize_linear_costs:
            inputs = self.adapter.virtual_features(
                self.t, self.u, x[: self.legacy_dimension], prepared=self.prepared
            )
            costs = self.adapter.cost_original.clone()
            new_linear = costs[:, 1].index_copy(
                0, self.cost_generators, x[self.legacy_dimension :]
            )
            costs = torch.stack((costs[:, 0], new_linear, costs[:, 2]), dim=1)
            gen = inputs["x_dict"]["gen"].clone()
            gen[:, C0_H : C2_H + 1] = signed_log_cost(costs)[
                self.prepared["active_generator_tensor"]
            ]
            inputs["x_dict"]["gen"] = gen
            inputs["virtual_costs"] = costs
            data = self.prepared["data"]
            prediction = self.adapter.model(
                inputs["x_dict"],
                data.edge_index_dict,
                inputs["edge_attr_dict"],
                data.mask_dict,
            )
            state = self.adapter._decode_state(
                prediction["bus"], prediction["gen"], inputs
            )
        else:
            state = self.adapter.forward(self.t, self.u, x, prepared=self.prepared)
        eq, ineq = self.constraints(state)
        return x, state, eq, ineq

    def constraints(self, state):
        """Original-network constraints as functions of the electrical state.

        Virtual model inputs affect these constraints only through pg, qg, vm,
        and va. Online indicators and original network data are fixed here.
        """
        raw = self.physics.evaluate(state, self.t)
        # The decoder fixes the REF angle; no independent reference-angle residual is needed.
        eq = raw["equality"][: 2 * self.physics.nb]
        ineq = raw["inequality"]
        if self.convention == "literal_period":
            # Replace native angle limits by the polar principal-atan domain.
            count = int(self.physics.has_amin.sum() + self.physics.has_amax.sum())
            if count:
                ineq = ineq[:-count]
            delta = raw["quantities"]["angle_difference_rad"]
            bound = np.pi / 2 - 1e-7
            ineq = torch.cat((ineq, -state["qg"] / 100, delta - bound, -delta - bound))
        return eq, ineq

    def jacobian(self, value, state, tensor):
        def check_deadline():
            if time.monotonic() >= getattr(self, "deadline", float("inf")):
                raise TimeoutError(
                    "start or overall budget reached during derivative evaluation"
                )

        check_deadline()
        if not tensor.numel():
            return np.zeros((len(value), 0))
        if self.jacobian_mode == "rowwise":
            # Retained as an independently evaluated derivative reference.
            rows = []
            for v in value:
                check_deadline()
                rows.append(torch.autograd.grad(v, tensor, retain_graph=True)[0])
            result = torch.stack(rows)
        else:
            keys = ("pg", "qg", "vm", "va")
            electrical = torch.cat([state[k] for k in keys])
            neural_jac = torch.autograd.grad(
                electrical,
                tensor,
                grad_outputs=torch.eye(
                    len(electrical), dtype=tensor.dtype, device=tensor.device
                ),
                is_grads_batched=True,
            )[0]
            check_deadline()
            # Detach before differentiating physics: qg, pg, vm and va need to
            # be independent coordinates in this partial Jacobian, even when
            # the released decoder computes one from another.
            independent = electrical.detach().requires_grad_(True)
            parts = independent.split([state[k].numel() for k in keys])
            physical_state = dict(zip(keys, parts), uc=state["uc"].detach())
            eq, ineq = self.constraints(physical_state)
            physical_values = torch.cat(
                (eq, ineq, physical_state["pg"] / self.physics.base)
            )
            physical_jac = torch.autograd.grad(
                physical_values,
                independent,
                grad_outputs=torch.eye(
                    len(physical_values), dtype=tensor.dtype, device=tensor.device
                ),
                is_grads_batched=True,
            )[0]
            result = physical_jac @ neural_jac
        check_deadline()
        return result.detach().numpy()

    def evaluate(self, z, derivatives=False):
        z = np.asarray(z, float)
        if z.shape != (len(self.free),) or not np.isfinite(z).all():
            raise ValueError("Invalid period input coordinates")
        if self.cache is not None and np.array_equal(self.cache["z"], z):
            if not derivatives or "jac" in self.cache:
                return self.cache
        tensor = torch.tensor(z, dtype=torch.float64, requires_grad=derivatives)
        x, state, eq, ineq = self.forward(tensor)
        value = torch.cat((eq, ineq, state["pg"] / self.physics.base))
        if not torch.isfinite(value).all():
            raise FloatingPointError("Nonfinite period constraints")
        result = dict(
            z=z.copy(),
            x=x.detach().numpy(),
            state={
                k: state[k].detach().numpy() for k in ["uc", "pg", "qg", "vm", "va"]
            },
            eq=eq.detach().numpy(),
            ineq=ineq.detach().numpy(),
        )
        if derivatives:
            all_jac = self.jacobian(value, state, tensor)
            if not np.isfinite(all_jac).all():
                raise FloatingPointError("Nonfinite period Jacobian")
            result["jac"] = all_jac[: len(eq) + len(ineq)]
            result["interface_jac"] = np.vstack(
                (np.zeros((len(self.u), len(z))), all_jac[len(eq) + len(ineq) :])
            )
        result["interface"] = np.r_[
            self.u, state["pg"].detach().numpy() / self.physics.base
        ]
        # Cached arrays are immutable; callers cannot corrupt later evaluations.
        for value in list(result.values()) + list(result["state"].values()):
            if isinstance(value, np.ndarray):
                value.setflags(write=False)
        self.calls += 1
        self.cache = result
        return result

    def check(self, result):
        c = self.adapter.adapter.static_physical_case(self.t, self.u)
        if self.convention == "literal_period":
            for g in c["gen"]:
                g["min_q_mvar"] = max(0.0, g["min_q_mvar"])
            for e in c["branch"]:
                e["ang_min"] = np.rad2deg(-np.pi / 2 + 1e-7)
                e["ang_max"] = np.rad2deg(np.pi / 2 - 1e-7)
        solution = {k: v[None, :] for k, v in result["state"].items()}
        checked = check_solution(
            c,
            solution,
            power_tolerance=1e-3,
            voltage_tolerance=1e-5,
            binary_tolerance=1e-9,
            angle_tolerance=1e-6,
        )
        exact_binary = bool(np.isin(solution["uc"], [0.0, 1.0]).all())
        bound_error = float(
            max(
                np.max(self.lo.numpy() - result["x"]),
                np.max(result["x"] - self.hi.numpy()),
                0.0,
            )
        )
        atan_error = 0.0
        angle_domain = True
        if self.convention == "literal_period":
            d = ac_quantities(c, solution["vm"], solution["va"])
            delta = d["angle_difference_rad"][:, d["active"]]
            angle_domain = bool((np.abs(delta) < np.pi / 2).all())
            atan_error = float(
                np.max(np.abs(delta - np.arctan(np.tan(delta))), initial=0.0)
            )
        # Reactive generation follows the adapter's bus-total allocation.
        return dict(
            accepted_period=bool(
                checked["feasible"]
                and exact_binary
                and bound_error <= 1e-12
                and angle_domain
                and atan_error <= 1e-6
            ),
            physical_check=checked,
            exact_binary=exact_binary,
            input_bound_violation=bound_error,
            principal_atan_residual=atan_error,
            principal_angle_domain=angle_domain,
            neural_state_obtained_by_substitution=True,
            scope="single-period neural-image feasibility only; no temporal UC claim",
        )
