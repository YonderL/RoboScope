"""Derive the official four-policy table from audited, versioned episode sources."""

import csv
import hashlib
import json
from pathlib import Path

from roboscope.reporting.evaluation import load_results
from roboscope.reporting.native import audit_snapshot
from roboscope.reporting.records import audit_portable

POLICIES = ("act", "dp", "smolvla", "pi0")
LABELS = ("ACT", "DP", "SmolVLA", "Pi-0 LoRA")
# Native LIBERO-Spatial order -> sorted HDF5/HF manifest order. Match task
# names, never episode identities; native episode indices remain native.
NATIVE_TO_MANIFEST = (0, 5, 1, 6, 2, 7, 3, 8, 4, 9)
TASK_LABELS = (
    "Between plate and ramekin",
    "Table center",
    "Top drawer",
    "Next to cookie box",
    "Next to plate",
    "Next to ramekin",
    "On cookie box",
    "On the ramekin",
    "On the stove",
    "On the wooden cabinet",
)


def build_summary(results):
    results = Path(results)
    baseline = results / "libero_spatial"
    with (baseline / "episodes.csv").open() as handle:
        historical = audit_portable(list(csv.DictReader(handle)))
    provenance = json.loads((baseline / "provenance.json").read_text())
    tasks = json.loads((baseline / "protocol.json").read_text())["tasks"]
    names = {t["id"]: t["name"] for t in tasks}
    if set(names) != set(range(10)) or len(set(names.values())) != 10:
        raise ValueError("Expected ten unique task names")
    snapshot = audit_snapshot(json.loads((results / "vla_spatial/snapshot.json").read_text()))
    candidates = [s for s in snapshot["evaluations"] if s["name"] == "smolvla_100k_exec10_50ep"]
    if len(candidates) != 1:
        raise ValueError("Expected one final native SmolVLA study")
    smol = candidates[0]
    protocol, followups = load_results(results / "mujoco332")
    if smol["checkpoint_sha256"] != protocol["studies"]["smolvla"]["checkpoint_sha256"]:
        raise ValueError("SmolVLA replacement uses a different checkpoint")
    for policy in ("act", "dp"):
        old = provenance[f"{policy}_final"]
        new = protocol["studies"][policy]
        if old["gradient_step"] != new["gradient_step"]:
            raise ValueError("Replacement uses a different training step")
        if any(r["checkpoint_sha256"] != old["checkpoint_sha256"] for r in historical[f"{policy}_final"]):
            raise ValueError("Historical checkpoint disagrees with provenance")
    if provenance["dp_final"]["checkpoint_sha256"] != protocol["studies"]["dp"]["checkpoint_sha256"]:
        raise ValueError("DP replacement uses a different checkpoint")

    rows = []
    for task_id in range(10):
        native_id = NATIVE_TO_MANIFEST.index(task_id)
        for policy in POLICIES:
            replacement = policy == "pi0" or task_id == 7
            if replacement:
                source_id = native_id if policy == "smolvla" else task_id
                selected = [r for r in followups[policy] if int(r["task_id"]) == source_id]
                checkpoint = protocol["studies"][policy]["checkpoint_sha256"]
                source = "mujoco332/episodes.csv"
                identity = "native_episode_index" if policy == "smolvla" else "fixed_initial_state"
            elif policy == "smolvla":
                source_id = native_id
                selected = [r for r in smol["episodes"] if r["task_id"] == native_id]
                checkpoint = smol["checkpoint_sha256"]
                source = "vla_spatial/snapshot.json"
                identity = "native_episode_index"
            else:
                source_id = task_id
                selected = [r for r in historical[f"{policy}_final"] if int(r["task_id"]) == task_id]
                checkpoint = provenance[f"{policy}_final"]["checkpoint_sha256"]
                source = "libero_spatial/episodes.csv"
                identity = "fixed_initial_state"
            if len(selected) != 50:
                raise ValueError("Expected exactly 50 trials per task and policy")
            if any(r.get("task_name", names[task_id]) != names[task_id] for r in selected):
                raise ValueError("Task name disagrees with manifest")
            successes = sum(int(r["success"]) for r in selected)
            rows.append(
                {
                    "task_id": task_id,
                    "task_name": names[task_id],
                    "policy": policy,
                    "successes": successes,
                    "episodes": 50,
                    "success_rate_percent": 2 * successes,
                    "source": source,
                    "source_task_id": source_id,
                    "identity_kind": identity,
                    "checkpoint_sha256": checkpoint,
                }
            )
    totals = {}
    for policy in POLICIES:
        successes = sum(r["successes"] for r in rows if r["policy"] == policy)
        totals[policy] = {"successes": successes, "episodes": 500, "success_rate_percent": successes / 5}
    sources = (
        "libero_spatial/episodes.csv",
        "libero_spatial/protocol.json",
        "libero_spatial/provenance.json",
        "vla_spatial/snapshot.json",
        "mujoco332/episodes.csv",
        "mujoco332/protocol.json",
    )
    return {
        "schema": 1,
        "date": "2026-09-28",
        "suite": "libero_spatial",
        "selection": "Replace only on-ramekin ACT/DP/SmolVLA with the completed MuJoCo 3.3.2 reruns; Pi-0 uses its full MuJoCo 3.3.2 evaluation.",
        "training_seed": 0,
        "episodes_per_task": 50,
        "native_to_manifest_task_ids": list(NATIVE_TO_MANIFEST),
        "source_sha256": {s: hashlib.sha256((results / s).read_bytes()).hexdigest() for s in sources},
        "notes": [
            "Historical archives are preserved; this table is the official selection derived from them.",
            "Native SmolVLA episode positions are not fixed initial-state IDs and are not paired with ACT/DP/Pi-0.",
            "ACT historical and rerun checkpoint files have different fingerprints; each source retains its recorded identity and 56064-step label.",
            "Observations and rollout budgets differ between policy recipes. One training seed does not establish a stable ranking.",
        ],
        "totals": totals,
        "per_task": rows,
    }


def render(data, output):
    from roboscope.reporting.plotting import plt, save, style

    summary = build_summary(data)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), width_ratios=(2.1, 1), layout="constrained")
    cells = [[r for r in summary["per_task"] if r["task_id"] == t] for t in range(10)]
    values = [[r["success_rate_percent"] for r in row] for row in cells]
    heat = axes[0].imshow(values, vmin=0, vmax=100, cmap="YlGnBu", aspect="auto")
    axes[0].grid(False)
    axes[0].set(
        xticks=range(4),
        xticklabels=LABELS,
        yticks=range(10),
        yticklabels=TASK_LABELS,
        title="Per-task SR · 50 trials per cell",
    )
    axes[0].tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
    for task_id, row in enumerate(cells):
        for column, r in enumerate(row):
            axes[0].text(
                column,
                task_id,
                f"{r['success_rate_percent']}%\n{r['successes']}/50",
                ha="center",
                va="center",
                color="white" if r["success_rate_percent"] >= 70 else "#172B3A",
                fontsize=10,
            )
    fig.colorbar(heat, ax=axes[0], label="Success rate (%)", shrink=0.7)
    totals = [summary["totals"][p] for p in POLICIES]
    bars = axes[1].bar(
        range(4),
        [t["success_rate_percent"] for t in totals],
        color=["#2764A5", "#008675", "#D08B32", "#8A5AA6"],
        width=0.65,
    )
    axes[1].bar_label(
        bars,
        labels=[f"{t['success_rate_percent']:.1f}%\n{t['successes']}/500" for t in totals],
        padding=5,
        fontsize=10,
    )
    axes[1].set(
        title="Full suite · 500 trials per policy",
        xticks=range(4),
        xticklabels=LABELS,
        ylim=(0, 112),
        ylabel="Success rate (%)",
    )
    axes[1].tick_params(axis="x", labelrotation=25)
    fig.suptitle("LIBERO-Spatial | official results · seed 0", fontsize=17, fontweight="bold")
    fig.supxlabel(
        "ACT/DP/SmolVLA on-ramekin: MuJoCo 3.3.2 reruns; other nine tasks: archived runs.\n"
        "Pi-0 LoRA: all ten tasks on MuJoCo 3.3.2. Observation and rollout protocols differ.",
        fontsize=9,
    )
    save(fig, output, "policy_success_rates")
    return summary
