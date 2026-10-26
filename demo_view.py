"""Cost, commitment and AC balance plots from the current solve."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from literal_check import check


def residuals(case, best):
    witness = {key: best["horizon"][key] for key in ("u", "su", "sd")}
    result = []
    for state in ("raw_prediction", "state"):
        witness.update(
            {
                key: [period[state][key] for period in best["periods"]]
                for key in ("pg", "qg", "vm", "va")
            }
        )
        violations = check(case, witness)["max_violations"]
        result.append([violations["p_balance_mw"], violations["q_balance_mvar"]])
    return result


def render(report_path, destination):
    report = json.loads(Path(report_path).read_text())
    best = report.get("best")
    if best is None:
        return False
    root = Path(__file__).resolve().parent
    case = json.loads((root / report["configuration"]["case"]).read_text())
    history = report["accepted_history"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="constrained")
    color = "#176a91"
    axes[0].step(
        [r["elapsed_s"] for r in history],
        [r["best_objective"] for r in history],
        where="post",
        color=color,
        marker="o",
        markersize=3,
    )
    axes[0].set(
        title="Best feasible cost",
        xlabel="Computation time (s)",
        ylabel="Operating cost",
    )
    axes[0].ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[0].grid(alpha=0.2)
    online = np.asarray(best["horizon"]["u"]).T
    axes[1].imshow(
        online, cmap="Blues", vmin=0, vmax=1, aspect="auto", interpolation="nearest"
    )
    axes[1].set(
        title="Commitment schedule",
        xlabel="Period",
        ylabel="Generator",
        xticks=range(online.shape[1]),
        xticklabels=range(1, online.shape[1] + 1),
        yticks=range(online.shape[0]),
        yticklabels=range(1, online.shape[0] + 1),
    )
    for (g, t), value in np.ndenumerate(online):
        axes[1].text(
            t,
            g,
            str(int(value)),
            ha="center",
            va="center",
            color="white" if value else "#333333",
        )
    before, after = residuals(case, best)
    positions = np.arange(2)
    axes[2].bar(
        positions - 0.18,
        np.maximum(before, 1e-12),
        0.36,
        label="GENCO prediction",
        color="#b97930",
    )
    axes[2].bar(
        positions + 0.18,
        np.maximum(after, 1e-12),
        0.36,
        label="Corrected output",
        color=color,
    )
    axes[2].axhline(1e-3, color="#444444", ls="--", lw=1, label="Tolerance")
    axes[2].set(
        yscale="log",
        title="Maximum AC balance residual",
        xticks=positions,
        xticklabels=["Real (MW)", "Reactive (MVAr)"],
    )
    axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(destination, format="svg", metadata={"Date": None})
    plt.close(fig)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("destination")
    args = parser.parse_args()
    render(args.report, args.destination)
