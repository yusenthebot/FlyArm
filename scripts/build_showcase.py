"""Build the overnight results page (HTML + videos + UI screenshots) from run artifacts.

Every number on the page is read from a results.json file; missing or running experiments
are shown as such. Usage: uv run python scripts/build_showcase.py runs/overnight/site
"""

from __future__ import annotations

import html
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
VIDEOS = RUNS / "overnight" / "videos"
SHOTS = RUNS / "overnight" / "screenshots"

KITCHEN = {
    "flyleg": "Fly CNS · leg proprioceptors + head senses",
    "flyleg_shuffled": "Shuffled CNS · same interface",
    "mlp": "MLP behavior cloning",
    "gru": "GRU · parameter-matched",
}
PROPRIO = {
    "flyleg": "Fly CNS · leg proprioceptors only",
    "flyleg_shuffled": "Shuffled CNS · proprioceptors only",
}
BRAIN = {"connectome": "Complete MaleCNS", "shuffled": "Shuffled CNS", "gru": "GRU · matched"}


def load(name: str) -> dict[str, Any] | None:
    path = RUNS / name / "results.json"
    return json.loads(path.read_text()) if path.is_file() else None


def esc(value: object) -> str:
    return html.escape(str(value))


def mean_sd(values: list[float], digits: int = 1) -> str:
    if not values:
        return "&mdash;"
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{np.mean(values):.{digits}f}<small> ± {np.std(values, ddof=1):.{digits}f}</small>"


def status(results: dict[str, Any] | None) -> str:
    if results is None:
        return '<span class="pill pending">not started</span>'
    state = results.get("status", "unknown")
    return f'<span class="pill {esc(state)}">{esc(state)}</span>'


def kitchen_table(results: dict[str, Any] | None, names: dict[str, str]) -> str:
    if results is None or not results.get("models"):
        return '<p class="empty">No completed checkpoints yet.</p>'
    models = results["models"]
    tasks = models[0]["clean"]["tasks"]
    variants = list(models[0]["ood"])
    head = "".join(f"<th>{esc(task)}</th>" for task in tasks)
    ood_head = "".join(f"<th>{esc(v.replace('_', ' '))}</th>" for v in variants)
    rows = []
    for kind, label in names.items():
        group = [model for model in models if model["kind"] == kind]
        if not group:
            continue
        clean = [model["clean"]["normalized_score"] for model in group]
        tasks_cells = "".join(
            f"<td>{100 * np.mean([m['clean']['per_task_success'][t] for m in group]):.0f}%</td>"
            for t in tasks
        )
        ood_cells = "".join(
            f"<td>{mean_sd([m['ood'][v]['normalized_score'] for m in group])}</td>"
            for v in variants
        )
        seeds = ", ".join(f"{value:.1f}" for value in clean)
        rows.append(
            f"<tr><th scope='row'>{esc(label)}</th>"
            f"<td>{group[0]['trainable_parameters']:,}</td><td class='score'>{mean_sd(clean)}</td>"
            f"<td class='seeds'>{seeds}</td>{tasks_cells}{ood_cells}</tr>"
        )
    return (
        "<div class='table-wrap'><table><thead><tr><th>Controller</th><th>Params</th>"
        "<th>Score</th><th>Per seed</th>"
        f"{head}{ood_head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def lesion_table(results: dict[str, Any] | None) -> str:
    if results is None:
        return ""
    fly = [m for m in results.get("models", []) if m["kind"] == "flyleg" and "lesions" in m]
    if not fly:
        return ""
    names = list(fly[0]["lesions"])
    head = "".join(f"<th>{esc(name.replace('_', ' '))}</th>" for name in names)
    cells = "".join(
        f"<td>{mean_sd([m['lesions'][name]['normalized_score'] for m in fly])}</td>"
        for name in names
    )
    intact = mean_sd([m["clean"]["normalized_score"] for m in fly])
    return (
        "<div class='table-wrap'><table class='lesions'><thead><tr><th>Intact</th>"
        f"{head}</tr></thead><tbody><tr><td class='score'>{intact}</td>{cells}</tr></tbody>"
        "</table></div>"
    )


def brain_table(runs: list[str], fields: list[tuple[str, str]]) -> str:
    models = [m for name in runs if (r := load(name)) for m in r.get("models", [])]
    if not models:
        return '<p class="empty">No completed checkpoints yet.</p>'
    head = "".join(f"<th>{esc(label)}</th>" for _, label in fields)
    rows = []
    for kind, label in BRAIN.items():
        group = [m for m in models if m["kind"] == kind]
        if not group:
            continue
        cells = []
        for field, _ in fields:
            if field == "mean_final_distance_m":
                value = 1000 * np.mean([m["clean"][field] for m in group])
                cells.append(f"<td>{value:.2f} mm</td>")
                continue
            episodes = len(group[0]["clean"]["episodes"])
            counts = [round(m["clean"][field] * episodes) for m in group]
            cells.append(
                f"<td><b>{sum(counts)}</b>/{episodes * len(group)}"
                f"<small> ({', '.join(map(str, counts))})</small></td>"
            )
        rows.append(
            f"<tr><th scope='row'>{esc(label)}</th><td>{len(group)}</td>{''.join(cells)}</tr>"
        )
    brain = [m for m in models if m["kind"] == "connectome"]
    ablations = [a for a in ("edges_off", "direct_only", "state_reset_every_step") if a in brain[0]]
    for ablation in ablations:
        cells = []
        for field, _ in fields:
            if field == "mean_final_distance_m":
                value = 1000 * np.mean([m[ablation][field] for m in brain])
                cells.append(f"<td>{value:.1f} mm</td>")
                continue
            episodes = len(brain[0][ablation]["episodes"])
            total = sum(round(m[ablation][field] * episodes) for m in brain)
            cells.append(f"<td>{total}/{episodes * len(brain)}</td>")
        rows.append(
            f"<tr class='ablation'><th scope='row'>MaleCNS · {esc(ablation.replace('_', ' '))}"
            f"</th><td>{len(brain)}</td>{''.join(cells)}</tr>"
        )
    return (
        f"<div class='table-wrap'><table><thead><tr><th>Controller</th><th>Seeds</th>{head}"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def video(site: Path, name: str, caption: str, wide: bool = False) -> str:
    source = VIDEOS / name
    if not source.is_file():
        return ""
    target = site / "videos" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    cls = "clip wide" if wide else "clip"
    return (
        f"<figure class='{cls}'><video src='videos/{esc(name)}' controls muted playsinline "
        f"preload='metadata'></video><figcaption>{caption}</figcaption></figure>"
    )


def screenshot(site: Path, name: str, caption: str) -> str:
    source = SHOTS / name
    if not source.is_file():
        return ""
    target = site / "screenshots" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return (
        f"<figure class='shot'><img src='screenshots/{esc(name)}' alt='{esc(caption)}'>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def build(site: Path) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    complete = load("flyleg-kitchen-complete-001")
    proprio = load("flyleg-kitchen-complete-proprio-001")
    mixed = load("flyleg-kitchen-mixed-001")
    template = (Path(__file__).parent / "showcase_template.html").read_text()
    kitchen_videos = "".join(
        [
            video(
                site,
                "kitchen-complete-comparison-seed0-episode0.mp4",
                "Same held-out episode, four controllers trained on the same 17 human "
                "demonstrations (seed 0).",
                wide=True,
            ),
            video(
                site,
                "kitchen-complete-lesions-seed0-episode0.mp4",
                "Causal lesions of the trained fly controller on the same episode: intact, "
                "head senses removed, leg proprioceptors silenced, every edge removed.",
                wide=True,
            ),
            video(
                site,
                "proprio/kitchen-complete-flyleg-seed0.mp4",
                "Fly CNS wired through leg proprioceptors only, three episodes.",
            ),
            video(site, "kitchen-complete-flyleg-seed1.mp4", "Fly CNS, training seed 1."),
            video(site, "kitchen-complete-mlp-seed0.mp4", "MLP behavior cloning, three episodes."),
            video(site, "kitchen-complete-gru-seed0.mp4", "GRU, three episodes."),
        ]
    )
    pick_videos = "".join(
        video(
            site,
            f"pick-place-{kind}-seed{seed}.mp4",
            f"{BRAIN[kind]} · seed {seed} · three held-out episodes.",
        )
        for kind in ("connectome", "shuffled", "gru")
        for seed in (0, 1, 2)
    )
    shots = "".join(
        [
            screenshot(
                site,
                "ui-kitchen.png",
                "Kitchen live view: the complete CNS with the leg "
                "interface in color, the Franka in FrankaKitchen, per-task distances.",
            ),
            screenshot(
                site,
                "ui-pick-place.png",
                "Pick-and-place live view with the Menagerie "
                "Franka meshes and the 140k-neuron soma map.",
            ),
        ]
    )
    notes_path = RUNS / "overnight" / "findings.json"
    notes = json.loads(notes_path.read_text()) if notes_path.is_file() else {}
    findings = "".join(
        f"<article class='finding'><span class='tag {esc(item['class'])}'>{esc(item['tag'])}</span>"
        f"<h3>{esc(item['title'])}</h3><p>{item['text']}</p></article>"
        for item in notes.get("findings", [])
    )
    caveats = "".join(f"<li>{item}</li>" for item in notes.get("caveats", []))
    values = {
        "lede": notes.get("lede", "Results are filled in as runs complete."),
        "findings": findings,
        "caveats": caveats,
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "complete_status": status(complete),
        "proprio_status": status(proprio),
        "mixed_status": status(mixed),
        "kitchen_table": kitchen_table(complete, KITCHEN),
        "kitchen_lesions": lesion_table(complete),
        "proprio_table": kitchen_table(proprio, PROPRIO),
        "proprio_lesions": lesion_table(proprio),
        "mixed_table": kitchen_table(mixed, KITCHEN),
        "published": str(complete["published_bc_reference"]) if complete else "65.0",
        "kitchen_videos": kitchen_videos,
        "pick_table": brain_table(
            ["whole-brain-pick-place-001", "whole-brain-pick-place-002"],
            [("grasp_rate", "Grasp"), ("lift_rate", "Lift"), ("success_rate", "Stable place")],
        ),
        "pick_videos": pick_videos,
        "reach_table": brain_table(
            ["whole-brain-reach-001"],
            [("success_rate", "Success"), ("mean_final_distance_m", "Final error")],
        ),
        "reach_video": video(
            site, "reach-connectome-seed0.mp4", "Complete MaleCNS reaching four held-out targets."
        ),
        "shots": shots,
    }
    page = template
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    output = site / "index.html"
    output.write_text(page)
    return output


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else RUNS / "overnight" / "site"))
