"""Publication figures generated solely from audited, portable episode records."""

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from roboscope.reporting.plotting import save, style
from roboscope.reporting.records import audit_portable

COLORS = {"act_final": "#2764A5", "dp_final": "#008675", "dp_best": "#D08B32"}
LABELS = {"act_final": "ACT · final", "dp_final": "DP · final (30k)", "dp_best": "DP · min. val. MSE (7k)"}
TASKS = [
    "Between plate / ramekin",
    "Table center",
    "Inside drawer",
    "Next to cookie box",
    "Next to plate",
    "Next to ramekin",
    "On cookie box",
    "On ramekin",
    "On stove",
    "On cabinet",
]


def paired(a, b):
    """Same task/init pairs, stratified by task; CI is rollout resampling, NOT seed std."""
    aa = {(int(r["task_id"]), int(r["initial_state_id"])): int(r["success"]) for r in a}
    bb = {(int(r["task_id"]), int(r["initial_state_id"])): int(r["success"]) for r in b}
    if aa.keys() != bb.keys():
        raise ValueError("Unmatched episodes")
    delta = np.array([[bb[t, i] - aa[t, i] for i in range(50)] for t in range(10)])
    rng = np.random.default_rng(2026)
    indices = rng.integers(0, 50, (10000, 10, 50))
    draws = delta[np.arange(10)[None, :, None], indices].mean((1, 2)) * 100
    return dict(
        delta_pp=float(delta.mean() * 100),
        rescued=int((delta == 1).sum()),
        lost=int((delta == -1).sum()),
        paired_stratified_bootstrap_ci95_pp=np.percentile(draws, [2.5, 97.5]).tolist(),
        bootstrap_seed=2026,
    )


def render(data, output):
    data, output = Path(data), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    with (data / "episodes.csv").open() as h:
        groups = audit_portable(list(csv.DictReader(h)))
    metadata = json.loads((data / "provenance.json").read_text())
    keys = ["act_final", "dp_final", "dp_best"]
    rates = {k: np.mean([int(r["success"]) for r in v]) * 100 for k, v in groups.items()}
    per_task = {
        k: [
            np.mean([int(r["success"]) for r in groups[k] if int(r["task_id"]) == t]) * 100 for t in range(10)
        ]
        for k in keys
    }
    fig = plt.figure(figsize=(13.5, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.65], hspace=0.40, wspace=0.38)
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("A  |  Matched evaluation", loc="left", fontweight="bold")
    ys = np.arange(3)
    ax.barh(ys, [rates[k] for k in keys], color=[COLORS[k] for k in keys], height=0.57)
    ax.set_yticks(ys, [LABELS[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Success rate (%)")
    for y, k in zip(ys, keys):
        ax.text(rates[k] + 1, y, f"{rates[k]:.1f}%  ({round(rates[k] * 5)}/500)", va="center", fontsize=10)
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("B  |  Full chunk inference latency", loc="left", fontweight="bold")
    vals = [metadata[k]["latency_p50_ms"] for k in keys]
    ax.barh(ys, vals, color=[COLORS[k] for k in keys], height=0.57)
    ax.set_yticks(ys, [LABELS[k] for k in keys])
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.30)
    ax.set_xlabel("Latency p50 (ms) · batch = 1")
    for y, v in zip(ys, vals):
        ax.text(v + 1, y, f"{v:.1f} ms", va="center")
    ax = fig.add_subplot(gs[1, :])
    ax.set_title("C  |  Per-task success · 50 fixed initial states per task", loc="left", fontweight="bold")
    x = np.arange(10)
    width = 0.25
    for offset, k in zip([-width, 0, width], keys):
        bars = ax.bar(x + offset, per_task[k], width, label=LABELS[k], color=COLORS[k])
        ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x, [f"T{i}\n{v}" for i, v in enumerate(TASKS)], rotation=28, ha="right", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Success rate (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3, frameon=False)
    fig.suptitle(
        "LIBERO-Spatial  |  ACT vs Diffusion Policy",
        fontsize=20,
        fontweight="bold",
        x=0.05,
        ha="left",
        y=1.00,
    )
    fig.text(
        0.05,
        0.945,
        "One suite-conditioned model / seed 0 · same 500 task–init pairs · ACT K=8 chunk; DP DDIM=10, Ta=8",
        fontsize=10,
    )
    fig.text(
        0.05,
        -0.10,
        "Single training seed; no cross-seed error bars. Training budgets and visual encoders differ.\nLatency includes GPU preprocessing, full prediction and action D2H; excludes observation H2D, simulator and network.\nDP final was evaluated after observing the 7k result; this is exploratory checkpoint analysis.",
        fontsize=9,
        color="#586575",
    )
    save(fig, output, "act_vs_dp")
    history = json.loads((data / "dp_train_history.json").read_text())
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    ax = axs[0]
    steps = [r["step"] / 1000 for r in history]
    ax.plot(
        steps, [r["train_noise_mse"] for r in history], label="Online train · random crop", color="#8A99AA"
    )
    ax.plot(
        steps,
        [r["val_noise_mse_ema"] for r in history],
        label="EMA validation · center crop",
        color=COLORS["dp_best"],
    )
    ax.axvline(7, ls=":", color="#8A99AA")
    ax.set(
        xlabel="Gradient steps (thousands)", ylabel="Noise MSE", title="Denoising loss is not rollout success"
    )
    ax.legend(frameon=False, fontsize=9)
    ax = axs[1]
    ax.bar(
        ["Min. val. MSE\n7k steps", "Final\n30k steps"],
        [rates["dp_best"], rates["dp_final"]],
        color=[COLORS["dp_best"], COLORS["dp_final"]],
        width=0.5,
    )
    ax.set(ylim=(0, 100), ylabel="Success rate (%)", title="Same DDIM=10 / Ta=8 evaluation")
    for i, k in enumerate(["dp_best", "dp_final"]):
        ax.text(i, rates[k] + 2, f"{rates[k]:.1f}%", ha="center", fontweight="bold")
    save(fig, output, "checkpoint_selection")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, ids, xvalues, title, xlabel in [
        (
            axs[0],
            ["dp_best_dp_ddim05_ta08", "dp_best", "dp_best_dp_ddim20_ta08"],
            [5, 10, 20],
            "DDIM steps · Ta=8",
            "Denoising steps",
        ),
        (
            axs[1],
            ["dp_best_dp_ddim10_ta01", "dp_best_dp_ddim10_ta04", "dp_best"],
            [1, 4, 8],
            "Execution horizon · DDIM=10",
            "Executed actions before replanning",
        ),
    ]:
        ax.plot(xvalues, [rates[k] for k in ids], "o-", color=COLORS["dp_best"])
        ax.set(ylim=(0, 100), xticks=xvalues, xlabel=xlabel, ylabel="Success rate (%)", title=title)
        for x, k in zip(xvalues, ids):
            ax.annotate(
                f"{rates[k]:.1f}%", (x, rates[k]), xytext=(0, 8), textcoords="offset points", ha="center"
            )
    fig.suptitle("Exploratory inference ablations · 7k checkpoint only (not DP final)", fontweight="bold")
    save(fig, output, "dp_best_ablations")
    with (data / "act_learning_curve.csv").open() as h:
        curves = list(csv.DictReader(h))
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.7), sharey=True, layout="constrained")
    for ax, mode in zip(axs, ["chunk", "replan", "ensemble"]):
        for k, color in zip([1, 8, 16, 32], ["#8995A3", "#2764A5", "#008675", "#D08B32"]):
            rows = sorted(
                [r for r in curves if int(r["chunk"]) == k and r["mode"] == mode],
                key=lambda r: int(r["epoch"]),
            )
            x = [int(r["epoch"]) for r in rows]
            y = np.array([float(r["mean"]) for r in rows])
            s = np.array([float(r["std"]) for r in rows])
            ax.plot(x, y, "o-", color=color, label=f"K={k}", ms=4)
            ax.fill_between(x, np.maximum(0, y - s), np.minimum(100, y + s), alpha=0.12, color=color)
        ax.set(
            title=mode if mode != "ensemble" else "Temporal ensemble",
            xlabel="Epoch",
            xticks=[8, 16, 24, 32],
            ylim=(0, 100),
        )
    axs[0].set_ylabel("Success rate (%)")
    axs[-1].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "Historical ACT ablations · 3 seeds, mean ± sample std · 20 episodes/task", fontweight="bold"
    )
    save(fig, output, "act_learning_curves")
    summary = {
        "success_rates_percent": rates,
        "dp_best_to_final": paired(groups["dp_best"], groups["dp_final"]),
        "act_to_dp_final": paired(groups["act_final"], groups["dp_final"]),
        "uncertainty_note": "Paired bootstrap resamples initial states within each task, conditional on these trained models. Not training-seed uncertainty.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
