"""Differentiable original-network constraints for frozen-image optimization.

The final feasibility certificate remains physical_check.py. This module never
reads the neural network's virtual demands, costs or branch admittances.
"""

from __future__ import annotations
import copy
import numpy as np
import torch


class TorchAC:
    def __init__(self, case, dtype=torch.float64):
        self.case = copy.deepcopy(case)
        self.dtype = dtype
        self.nb, self.ng = len(case["bus"]), len(case["gen"])
        if not np.isin([e.get("br_status", 1) for e in case["branch"]], [0, 1]).all():
            raise ValueError("Branch status must be binary")
        self.active = [
            j for j, e in enumerate(case["branch"]) if e.get("br_status", 1) == 1
        ]
        edges = [case["branch"][j] for j in self.active]
        ids = {int(b["bus"]): i for i, b in enumerate(case["bus"])}
        self.f = torch.tensor(
            [ids[int(e["from_bus"])] for e in edges], dtype=torch.long
        )
        self.t = torch.tensor([ids[int(e["to_bus"])] for e in edges], dtype=torch.long)
        self.gb = torch.tensor(
            [ids[int(g["bus"])] for g in case["gen"]], dtype=torch.long
        )
        self.ref = torch.tensor(
            [i for i, b in enumerate(case["bus"]) if b.get("REF", 0)], dtype=torch.long
        )
        z = np.asarray([complex(e["r"], e["x"]) for e in edges])
        if np.any(z == 0):
            raise ValueError("In-service zero-impedance branch")
        ys = 1 / z
        charging = 0.5j * np.asarray([e["b"] for e in edges])
        tap = np.asarray([e["tap"] if e["tap"] else 1.0 for e in edges])
        if np.any(tap <= 0):
            raise ValueError("Tap ratios must be positive")
        a = tap * np.exp(1j * np.deg2rad([e.get("shift", 0.0) for e in edges]))
        cdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
        self.yff = torch.tensor((ys + charging) / np.abs(a) ** 2, dtype=cdtype)
        self.yft = torch.tensor(-ys / np.conj(a), dtype=cdtype)
        self.ytf = torch.tensor(-ys / a, dtype=cdtype)
        self.ytt = torch.tensor(ys + charging, dtype=cdtype)
        self.shunt = torch.tensor(
            [complex(b["GS"], -b["BS"]) for b in case["bus"]], dtype=cdtype
        )
        self.shunt *= float(case.get("shunt_baseMVA", 100.0))
        self.base = float(case["baseMVA"])
        self.pd, self.qd = self.tensor(case["pd"]), self.tensor(case["qd"])
        self.pmin, self.pmax, self.qmin, self.qmax = [
            self.tensor([g[k] for g in case["gen"]])
            for k in ("min_p_mw", "max_p_mw", "min_q_mvar", "max_q_mvar")
        ]
        self.vmin, self.vmax = [
            self.tensor([b[k] for b in case["bus"]]) for k in ("min_vm_pu", "max_vm_pu")
        ]
        self.rate = self.tensor([e["rate_a"] for e in edges])
        self.limited = self.rate > 0
        self.amin = self.tensor(np.deg2rad([e.get("ang_min", -360.0) for e in edges]))
        self.amax = self.tensor(np.deg2rad([e.get("ang_max", 360.0) for e in edges]))
        self.has_amin, self.has_amax = self.amin > -2 * np.pi, self.amax < 2 * np.pi
        self.c0, self.c1, self.c2 = [
            self.tensor([g[k] for g in case["gen"]])
            for k in ("cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2")
        ]

    def tensor(self, value):
        return torch.as_tensor(value, dtype=self.dtype)

    def quantities(self, vm, va):
        voltage = torch.polar(vm, va)
        vf, vt = voltage[..., self.f], voltage[..., self.t]
        sf = self.base * vf * torch.conj(self.yff * vf + self.yft * vt)
        st = self.base * vt * torch.conj(self.ytf * vf + self.ytt * vt)
        outgoing = (
            torch.zeros_like(voltage)
            .index_add(-1, self.f, sf)
            .index_add(-1, self.t, st)
        )
        outgoing = outgoing + vm.square() * self.shunt
        return {
            "from_mva": sf,
            "to_mva": st,
            "outgoing_mva": outgoing,
            "angle_difference_rad": va[..., self.f] - va[..., self.t],
        }

    def evaluate(self, state, period):
        """Return e=0 and g<=0, scaled in 100 MW/Mvar/MVA, 0.1 pu, radians.

        All constraints use the original physical case. Full P/Q balance is
        retained even if a model decoder enforces some rows for virtual inputs.
        This is a static block; the separate integer master enforces UC/ramping.
        """
        pg, qg, vm, va, u = (state[k] for k in ("pg", "qg", "vm", "va", "uc"))
        quantities = self.quantities(vm, va)
        pbus = torch.zeros_like(vm).index_add(-1, self.gb, pg)
        qbus = torch.zeros_like(vm).index_add(-1, self.gb, qg)
        rp = pbus - self.pd[period] - quantities["outgoing_mva"].real
        rq = qbus - self.qd[period] - quantities["outgoing_mva"].imag
        eq = torch.cat([rp / 100, rq / 100, va[self.ref]])
        da = quantities["angle_difference_rad"]
        inequality = torch.cat(
            [
                (self.pmin * u - pg) / 100,
                (pg - self.pmax * u) / 100,
                (self.qmin * u - qg) / 100,
                (qg - self.qmax * u) / 100,
                10 * (self.vmin - vm),
                10 * (vm - self.vmax),
                (
                    torch.abs(quantities["from_mva"][self.limited])
                    - self.rate[self.limited]
                )
                / 100,
                (
                    torch.abs(quantities["to_mva"][self.limited])
                    - self.rate[self.limited]
                )
                / 100,
                self.amin[self.has_amin] - da[self.has_amin],
                da[self.has_amax] - self.amax[self.has_amax],
            ]
        )
        h = torch.cat([u, pg / 100, qbus / 100, 10 * vm])
        cost = (self.c0 + self.c1 * pg + self.c2 * pg.square()).sum()
        return {
            "equality": eq,
            "inequality": inequality,
            "h": h,
            "generation_cost": cost,
            "active_balance_mw": rp,
            "reactive_balance_mvar": rq,
            "quantities": quantities,
        }
