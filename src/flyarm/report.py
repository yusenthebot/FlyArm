"""Aggregate completed FlyArm runs into one quantitative Markdown report.

Every number is read from a run's results.json; runs that are missing or not complete are
listed as such instead of being summarized.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

KITCHEN_NAMES = {
    "flyleg": "Fly CNS as left front leg (MaleCNS)",
    "flyleg_shuffled": "Shuffled CNS, same leg interface",
    "mlp": "MLP BC (D4RL architecture)",
    "gru": "GRU, parameter-matched",
}
BRAIN_NAMES = {
    "connectome": "Full MaleCNS",
    "shuffled": "Full shuffled CNS",
    "gru": "GRU, parameter-matched",
}


def _load(path: Path) -> dict[str, Any] | None:
    results = path / "results.json"
    if not results.is_file():
        return None
    payload = json.loads(results.read_text())
    return payload if isinstance(payload, dict) else None


def _mean_sd(values: list[float]) -> str:
    if not values:
        return "n/a"
    if len(values) == 1:
        return f"{values[0]:.1f}"
    return f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}"


def _channels(run: Path) -> str:
    config = json.loads((run / "config.json").read_text())
    if config.get("sensory_channels") == "proprioception":
        return "fly wired through leg proprioceptors only"
    return "fly wired through leg proprioceptors + head senses"


def kitchen_section(run: Path) -> list[str]:
    results = _load(run)
    if results is None:
        return [f"_{run}: not found_", ""]
    split = results["split"]
    lines = [
        f"### FrankaKitchen `{split}`, {_channels(run)} (`{run.name}`, "
        f"status: {results['status']})",
        "",
        f"D4RL normalized score (25 per completed target task, 0-100); mean ± sd over training "
        f"seeds; published BC reference {results['published_bc_reference']} (original D4RL v0).",
        "",
    ]
    models = results.get("models", [])
    if not models:
        return [*lines, "_no completed models yet_", ""]
    tasks = models[0]["clean"]["tasks"]
    ood_names = list(models[0]["ood"])
    header = ["Controller", "Params", "Seeds", "Clean", "Per seed", *tasks, *ood_names]
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for kind, name in KITCHEN_NAMES.items():
        group = [model for model in models if model["kind"] == kind]
        if not group:
            continue
        clean = [model["clean"]["normalized_score"] for model in group]
        per_task = [
            f"{100 * np.mean([m['clean']['per_task_success'][task] for m in group]):.0f}%"
            for task in tasks
        ]
        ood = [
            _mean_sd([m["ood"][variant]["normalized_score"] for m in group])
            for variant in ood_names
        ]
        row = [
            name,
            f"{group[0]['trainable_parameters']:,}",
            str(len(group)),
            _mean_sd(clean),
            ", ".join(f"{value:.1f}" for value in clean),
            *per_task,
            *ood,
        ]
        lines.append("| " + " | ".join(row) + " |")
    fly = [model for model in models if model["kind"] == "flyleg" and "lesions" in model]
    if fly:
        lines += [
            "",
            "Lesions of the trained fly controller (same checkpoints, clean evaluation):",
            "",
        ]
        lesions = list(fly[0]["lesions"])
        lines += ["| " + " | ".join(["Intact", *lesions]) + " |", "|" + "---|" * (len(lesions) + 1)]
        intact = _mean_sd([model["clean"]["normalized_score"] for model in fly])
        values = [
            _mean_sd([m["lesions"][name]["normalized_score"] for m in fly]) for name in lesions
        ]
        lines.append("| " + " | ".join([intact, *values]) + " |")
    zero = results.get("zero_action", {}).get("normalized_score")
    lines += ["", f"Zero action: {zero}.", ""]
    return lines


def brain_section(runs: list[Path], task: str) -> list[str]:
    models: list[dict[str, Any]] = []
    status = []
    for run in runs:
        results = _load(run)
        if results is None:
            status.append(f"{run.name}: missing")
            continue
        status.append(f"{run.name}: {results['status']}")
        models += results.get("models", [])
    lines = [f"### B1a {task} ({'; '.join(status)})", ""]
    if not models:
        return [*lines, "_no completed models_", ""]
    fields: tuple[str, ...]
    if task == "pick-place":
        header = ["Controller", "Params", "Seeds", "Grasp", "Lift", "Stable place"]
        fields = ("grasp_rate", "lift_rate", "success_rate")
    else:
        header = ["Controller", "Params", "Seeds", "Success", "Mean final error (mm)"]
        fields = ("success_rate", "mean_final_distance_m")
    lines += ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for kind, name in BRAIN_NAMES.items():
        group = [model for model in models if model["kind"] == kind]
        if not group:
            continue
        cells = []
        for field in fields:
            values = [model["clean"][field] for model in group]
            if field == "mean_final_distance_m":
                cells.append(f"{1000 * np.mean(values):.2f}")
            else:
                episodes = len(group[0]["clean"]["episodes"])
                counts = [round(value * episodes) for value in values]
                cells.append(
                    f"{sum(counts)}/{episodes * len(group)} ({', '.join(map(str, counts))})"
                )
        lines.append(
            "| "
            + " | ".join([name, f"{group[0]['trainable_parameters']:,}", str(len(group)), *cells])
            + " |"
        )
    brain = [model for model in models if model["kind"] == "connectome"]
    ablations = [
        key for key in ("edges_off", "direct_only", "state_reset_every_step") if key in brain[0]
    ]
    if brain and ablations:
        lines += [
            "",
            "Post-training checks of the full-MaleCNS checkpoints (successes over all seeds):",
            "",
        ]
        lines += [
            "| Ablation | "
            + " | ".join(f.replace("_rate", "") for f in fields if f.endswith("rate"))
            + " |"
        ]
        lines += ["|---|" + "---|" * len([f for f in fields if f.endswith("rate")])]
        for ablation in ablations:
            cells = []
            for field in fields:
                if not field.endswith("rate"):
                    continue
                episodes = len(brain[0][ablation]["episodes"])
                total = sum(round(model[ablation][field] * episodes) for model in brain)
                cells.append(f"{total}/{episodes * len(brain)}")
            lines.append(f"| {ablation} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def build_report(root: Path) -> str:
    runs = root / "runs"
    lines = [
        "# FlyArm quantitative report",
        "",
        "Generated from run artifacts; every number below is read from a results.json file.",
        "",
        "## B2: the Franka arm as the fly's left front leg (D4RL FrankaKitchen)",
        "",
    ]
    for name in (
        "flyleg-kitchen-complete-001",
        "flyleg-kitchen-complete-proprio-001",
        "flyleg-kitchen-mixed-001",
    ):
        lines += kitchen_section(runs / name)
    lines += ["## B1a: complete MaleCNS controller, FlyArm Panda tasks", ""]
    lines += brain_section(
        [runs / "whole-brain-pick-place-001", runs / "whole-brain-pick-place-002"], "pick-place"
    )
    lines += brain_section([runs / "whole-brain-reach-001"], "reach")
    return "\n".join(lines) + "\n"
