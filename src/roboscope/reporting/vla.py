"""Figures for the native VLA protocol, kept separate from the matched ACT/DP study."""

import json
from pathlib import Path

from roboscope.reporting.native import audit_snapshot
from roboscope.reporting.plotting import plt, save, style


def render(data, output):
    data, output = Path(data), Path(output)
    snapshot = audit_snapshot(json.loads((data / "snapshot.json").read_text()))
    output.mkdir(parents=True, exist_ok=True)
    style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for ax, key, title in zip(
        axes[:2], ("smolvla", "pi0"), ("SmolVLA · 100k updates", "Pi-0 LoRA · 30k updates")
    ):
        rows = snapshot["training"][key]["rows"]
        ax.plot(
            [r["step"] / 1000 for r in rows],
            [r["train_loss"] for r in rows],
            color="#2764A5",
            alpha=0.7,
            linewidth=1,
            label="Train (logged)",
        )
        val = [r for r in rows if r.get("validation_loss") is not None]
        if val:
            ax.plot(
                [r["step"] / 1000 for r in val],
                [r["validation_loss"] for r in val],
                color="#D08B32",
                marker=".",
                label="Held-out trajectories",
            )
        ax.set(title=title, xlabel="Optimizer steps (thousands)", ylabel="Loss (log scale)", yscale="log")
        ax.legend(frameon=False, fontsize=8)
    periodic = snapshot["periodic_evaluation"]
    axes[2].plot(
        [r["step"] / 1000 for r in periodic], [r["success_rate"] for r in periodic], "o-", color="#008675"
    )
    for row in periodic:
        axes[2].annotate(
            f"{row['success_rate']:.0f}%",
            (row["step"] / 1000, row["success_rate"]),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
        )
    axes[2].set(
        title="SmolVLA · periodic rollouts",
        xlabel="Checkpoint steps (thousands)",
        ylabel="Success (%) · 100 episodes / checkpoint",
        ylim=(0, 100),
    )
    fig.suptitle("Training evidence | LIBERO-Spatial · seed 0", fontsize=17, fontweight="bold")
    fig.supxlabel(
        "Different loss objectives / datasets; loss magnitudes are not a policy ranking. "
        "Periodic rollouts execute 1 action. Pi-0 rollout results are reported separately.",
        fontsize=9,
    )
    save(fig, output, "vla_training")

    studies = snapshot["evaluations"]
    fig, axes = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1, 2]}, layout="constrained"
    )
    summary = {}
    for i, (study, color) in enumerate(zip(studies, ("#92B5D4", "#2764A5"))):
        rows = study["episodes"]
        count, successes = len(rows), sum(r["success"] for r in rows)
        rate = successes / count * 100
        axes[0].bar(i, rate, color=color, width=0.55)
        axes[0].text(i, rate + 2, f"{rate:.1f}%\n{successes}/{count}", ha="center", fontsize=11)
        summary[study["name"]] = {
            "successes": successes,
            "episodes": count,
            "success_rate_percent": round(rate, 1),
        }
    axes[0].set(
        xticks=range(len(studies)),
        xticklabels=[f"{s['episodes_per_task']} trials/task" for s in studies],
        ylim=(0, 103),
        ylabel="Success rate (%)",
        title="Same 100k checkpoint · execute 10",
    )
    full = max(studies, key=lambda s: s["episodes_per_task"])
    rates = [
        100 * sum(r["success"] for r in full["episodes"] if r["task_id"] == t) / full["episodes_per_task"]
        for t in full["task_ids"]
    ]
    bars = axes[1].bar(range(10), rates, color=["#D08B32" if i == 5 else "#2764A5" for i in range(10)])
    axes[1].bar_label(bars, fmt="%.0f%%", fontsize=9, padding=3)
    axes[1].set(
        xticks=range(10),
        xticklabels=[f"T{i}" for i in range(10)],
        ylim=(0, 112),
        ylabel="Success rate (%)",
        title="Per-task outcome · 50 trials each",
    )
    fig.suptitle("SmolVLA | native 256px / 8D-state protocol", fontsize=17, fontweight="bold")
    fig.supxlabel(
        "Single training seed. Native episode indices are not audited ACT/DP initial-state IDs.\n"
        "MuJoCo 3.3.2 follow-up evaluations are reported separately; this historical aggregate is unchanged.",
        fontsize=9,
    )
    save(fig, output, "smolvla_evaluation")
    (output / "vla_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
