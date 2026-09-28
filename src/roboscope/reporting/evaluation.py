"""Audit and plot the completed MuJoCo 3.3.2 follow-up studies."""

import csv
import json
from pathlib import Path


def load_results(data):
    data = Path(data)
    protocol = json.loads((data / "protocol.json").read_text())
    if protocol["schema"] != 1:
        raise ValueError("Unsupported evaluation result schema")
    with (data / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    groups = {}
    for row in rows:
        if row["policy"] not in protocol["studies"]:
            raise ValueError("Unknown evaluation policy")
        groups.setdefault(row["policy"], []).append(row)
    if groups.keys() != protocol["studies"].keys():
        raise ValueError("Missing evaluation study")
    for name, study in protocol["studies"].items():
        selected = groups[name]
        expected = {
            (task, initial) for task in study["task_ids"] for initial in range(study["episodes_per_task"])
        }
        keys = {(int(r["task_id"]), int(r["trial_index"])) for r in selected}
        if keys != expected or len(selected) != len(keys):
            raise ValueError("Missing or duplicate evaluation trial")
        for row in selected:
            if row["success"] not in ("0", "1") or row["checkpoint_sha256"] != study["checkpoint_sha256"]:
                raise ValueError("Invalid outcome or mixed checkpoint")
            native = name == "smolvla"
            if row["identity_kind"] != ("native_episode_index" if native else "fixed_initial_state"):
                raise ValueError("Mismatched episode identity convention")
            if not native and int(row["eval_seed"]) != 10000 + 1000 * int(row["task_id"]) + int(
                row["trial_index"]
            ):
                raise ValueError("Invalid fixed-state seed")
    return protocol, groups


def render(data, output):
    from roboscope.reporting.plotting import plt, save, style

    protocol, groups = load_results(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    summary = {}
    for name, rows in groups.items():
        success = sum(int(r["success"]) for r in rows)
        summary[name] = {
            "episodes": len(rows),
            "successes": success,
            "success_rate_percent": 100 * success / len(rows),
        }
    fig, axes = plt.subplots(
        1, 2, figsize=(13.5, 5), gridspec_kw={"width_ratios": [2, 1]}, layout="constrained"
    )
    pi0 = groups["pi0"]
    rates = [
        100 * sum(int(r["success"]) for r in pi0 if int(r["task_id"]) == task) / 50 for task in range(10)
    ]
    bars = axes[0].bar(range(10), rates, color="#2764A5")
    axes[0].bar_label(bars, fmt="%.0f%%", fontsize=9, padding=3)
    total = summary["pi0"]
    axes[0].set(
        title=f"Pi-0 LoRA final · {total['successes']}/500 ({total['success_rate_percent']:.1f}%)",
        xticks=range(10),
        xticklabels=[f"T{i}" for i in range(10)],
        ylim=(0, 113),
        ylabel="Success (%) · 50 fixed initial states/task",
        xlabel="Task ID in training manifest",
    )
    names = ["act", "dp", "smolvla"]
    bars = axes[1].bar(
        range(3),
        [summary[n]["success_rate_percent"] for n in names],
        color=["#2764A5", "#008675", "#D08B32"],
        width=0.55,
    )
    axes[1].bar_label(bars, labels=[f"{summary[n]['successes']}/50" for n in names], padding=4)
    axes[1].set(
        title="Black bowl on ramekin · follow-up",
        xticks=range(3),
        xticklabels=["ACT", "DP", "SmolVLA"],
        ylim=(0, 113),
        ylabel="Success (%)",
        xlabel="Protocol differs across policy families",
    )
    fig.suptitle("MuJoCo 3.3.2 | RTX PRO 5000 · seed 0", fontsize=17, fontweight="bold")
    fig.supxlabel(
        "ACT/DP: 128px, 600-step limit · SmolVLA: 256px, 280 steps · Pi-0: 256px, 220 steps\n"
        "Native SmolVLA task 5 = manifest task 7. Single training seed; these are not controlled architecture comparisons.",
        fontsize=9,
    )
    save(fig, output, "mujoco332_evaluation")
    (output / "mujoco332_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
