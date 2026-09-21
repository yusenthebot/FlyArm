"""Build the milestone results page: kitchen protocol v2, the protocol sweep, and the
pick-and-place replication across seeds and shuffle replicates.

Every number is read from run artifacts (results.json files and the protocol sweep);
missing or running experiments are shown as such. Usage:

    PYTHONPATH=src uv run python scripts/build_milestone.py runs/milestone/site
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

from flyarm.whole_brain.stats import topology_statistics

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
MILESTONE = RUNS / "milestone"
KITCHEN_RUN = "flyleg-kitchen-complete-chunk-001"
OLD_KITCHEN_RUN = "flyleg-kitchen-complete-001"
PICK_PLACE_RUNS = [f"whole-brain-pick-place-00{index}" for index in range(1, 5)]

KITCHEN = {
    "flyleg": "Fly CNS · leg proprioceptors + head senses",
    "flyleg_shuffled": "Shuffled CNS · same interface",
    "gru": "GRU · parameter-matched",
    "mlp": "MLP · D4RL BC architecture",
    "act": "ACT reference · not a fly model",
}
SETTINGS = {
    "c1-mse-valloss": "Old protocol: MSE, single step, lowest validation loss",
    "c1-l1-valloss": "L1 loss",
    "c10-l1-valloss": "L1 + 10-step action chunks",
    "c1-l1-closedloop": "L1 + closed-loop checkpoint selection",
    "c10-l1-closedloop": "L1 + chunks + closed-loop selection (final)",
}


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


def pill(results: dict[str, Any] | None) -> str:
    state = "not started" if results is None else results.get("status", "unknown")
    css = {"not started": "pending"}.get(state, state)
    return f'<span class="pill {esc(css)}">{esc(state)}</span>'


def table(head: list[str], rows: list[str], css: str = "") -> str:
    cells = "".join(f"<th>{esc(label)}</th>" for label in head)
    return (
        f"<div class='table-wrap'><table class='{css}'><thead><tr>{cells}</tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def kitchen_table(results: dict[str, Any] | None) -> str:
    if results is None or not results.get("models"):
        return '<p class="empty">No completed checkpoints yet.</p>'
    models = results["models"]
    tasks = models[0]["clean"]["tasks"]
    variants = list(models[0]["ood"])
    rows = []
    teacher = results.get("teacher")
    for kind, label in KITCHEN.items():
        group = [model for model in models if model["kind"] == kind]
        if not group:
            continue
        clean = [model["clean"]["normalized_score"] for model in group]
        task_cells = "".join(
            f"<td>{100 * np.mean([m['clean']['per_task_success'][t] for m in group]):.0f}%</td>"
            for t in tasks
        )
        ood_cells = "".join(
            f"<td>{mean_sd([m['ood'][v]['normalized_score'] for m in group])}</td>"
            for v in variants
        )
        seeds = ", ".join(f"{value:.0f}" for value in clean)
        css = " class='reference'" if kind == "act" else ""
        rows.append(
            f"<tr{css}><th scope='row'>{esc(label)}</th>"
            f"<td>{group[0]['trainable_parameters']:,}</td><td class='score'>{mean_sd(clean)}</td>"
            f"<td class='seeds'>{seeds}</td>{task_cells}{ood_cells}</tr>"
        )
    if teacher:
        clean = teacher["clean"]
        blanks = "<td>&mdash;</td>" * len(variants)
        task_cells = "".join(f"<td>{100 * clean['per_task_success'][t]:.0f}%</td>" for t in tasks)
        rows.append(
            "<tr class='reference'><th scope='row'>Demonstration tracker · teacher, not "
            f"learned</th><td>0</td><td class='score'>{clean['normalized_score']:.0f}</td>"
            f"<td class='seeds'>&mdash;</td>{task_cells}{blanks}</tr>"
        )
    head = ["Controller", "Params", "Score", "Per seed", *tasks]
    head += [variant.replace("_", " ") for variant in variants]
    return table(head, rows)


def lesion_table(results: dict[str, Any] | None) -> str:
    if results is None:
        return ""
    fly = [m for m in results.get("models", []) if m["kind"] == "flyleg" and "lesions" in m]
    if not fly:
        return '<p class="empty">No trained fly checkpoint yet.</p>'
    names = list(fly[0]["lesions"])
    cells = "".join(
        f"<td>{mean_sd([m['lesions'][name]['normalized_score'] for m in fly])}</td>"
        for name in names
    )
    intact = mean_sd([m["clean"]["normalized_score"] for m in fly])
    head = ["Intact", *[name.replace("_", " ") for name in names]]
    return table(head, [f"<tr><td class='score'>{intact}</td>{cells}</tr>"], "lesions")


def sweep_table() -> str:
    path = RUNS / "protocol-sweep" / "results.json"
    if not path.is_file():
        return '<p class="empty">Protocol sweep not run yet.</p>'
    results = json.loads(path.read_text())
    rows = []
    for setting, label in SETTINGS.items():
        cells = []
        for kind in ("mlp", "gru"):
            scores = [r["score"] for r in results if r["setting"] == setting and r["kind"] == kind]
            per_seed = ", ".join(f"{score:.0f}" for score in scores)
            cells.append(
                f"<td class='score'>{mean_sd(scores)}</td><td class='seeds'>{per_seed}</td>"
            )
        rows.append(f"<tr><th scope='row'>{esc(label)}</th>{''.join(cells)}</tr>")
    return table(["Protocol", "MLP", "MLP per seed", "GRU", "GRU per seed"], rows)


def pick_place_tables() -> tuple[str, str, str]:
    runs = {name: load(name) for name in PICK_PLACE_RUNS}
    status = " ".join(f"<span><b>{esc(n)}</b> {pill(r)}</span>" for n, r in runs.items())
    models = [m for r in runs.values() if r for m in r.get("models", [])]
    if not models:
        return status, '<p class="empty">No completed checkpoints yet.</p>', ""
    stats = topology_statistics(models)
    rows = []
    for name, outcome in stats["outcomes"].items():
        per_seed = " · ".join(
            f"s{row['seed']} <b>{row['connectome']}</b> vs {'/'.join(map(str, row['shuffled']))}"
            for row in outcome["per_seed"]
        )
        pairs = outcome["episode_pairs"]
        rows.append(
            f"<tr><th scope='row'>{esc(name)}</th><td class='left'>{per_seed}</td>"
            f"<td>{outcome['seeds_measured_better']} / {outcome['seeds_measured_worse']}</td>"
            f"<td class='score'>{outcome['seed_level_p']:.3g}</td>"
            f"<td>{pairs['measured_only']} / {pairs['shuffled_only']}</td>"
            f"<td>{outcome['episode_level_p']:.2g}</td></tr>"
        )
    topology = table(
        [
            "Outcome",
            "Per seed: MaleCNS vs each shuffle (of 24)",
            "Seeds better / worse",
            "Seed-level p",
            "Episode pairs",
            "Episode-level p",
        ],
        rows,
    )
    totals = []
    for kind, label in (("connectome", "Complete MaleCNS"), ("shuffled", "Shuffled CNS")):
        group = [m for m in models if m["kind"] == kind and m["seed"] in stats["seeds"]]
        cells = []
        for field in ("grasp_rate", "lift_rate", "success_rate"):
            episodes = len(group[0]["clean"]["episodes"]) if group else 0
            total = sum(round(m["clean"][field] * episodes) for m in group)
            cells.append(f"<td><b>{total}</b>/{episodes * len(group)}</td>")
        totals.append(
            f"<tr><th scope='row'>{esc(label)}</th><td>{len(group)}</td>{''.join(cells)}</tr>"
        )
    summary = table(["Controller", "Checkpoints", "Grasp", "Lift", "Stable place"], totals)
    return status, summary, topology


def media(site: Path, source: Path, folder: str, caption: str, wide: bool = False) -> str:
    if not source.is_file():
        return ""
    target = site / folder / source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    if source.suffix == ".png":
        return (
            f"<figure class='shot'><img src='{folder}/{esc(source.name)}' alt='{esc(caption)}'>"
            f"<figcaption>{caption}</figcaption></figure>"
        )
    cls = "clip wide" if wide else "clip"
    return (
        f"<figure class='{cls}'><video src='{folder}/{esc(source.name)}' controls muted "
        f"playsinline preload='metadata'></video><figcaption>{caption}</figcaption></figure>"
    )


def build(site: Path) -> Path:
    site.mkdir(parents=True, exist_ok=True)
    kitchen = load(KITCHEN_RUN)
    videos = MILESTONE / "videos"
    chunk = videos / "kitchen-chunk"
    kitchen_videos = "".join(
        [
            media(
                site,
                videos / "kitchen-complete-teacher.mp4",
                "videos",
                "The demonstration tracker, built only from the 19 demonstrations (nearest "
                "demonstrated state plus a joint correction): what the data supports.",
            ),
            *(
                media(
                    site,
                    chunk / f"kitchen-complete-comparison-seed{seed}-episode0.mp4",
                    "videos",
                    f"Seed {seed}, the same held-out episode for every controller: fly CNS, "
                    "shuffled CNS, MLP, GRU, ACT reference.",
                    wide=True,
                )
                for seed in range(3)
            ),
            *(
                media(
                    site,
                    chunk / f"kitchen-complete-lesions-seed{seed}-episode0.mp4",
                    "videos",
                    f"Fly CNS seed {seed}: intact, head senses removed, leg deafferented, "
                    "every connectome edge removed.",
                    wide=True,
                )
                for seed in range(3)
            ),
        ]
    )
    pick_videos = "".join(
        media(site, videos / name, "videos", caption)
        for seed in (3, 4, 5)
        for name, caption in (
            (f"pick-place-connectome-seed{seed}.mp4", f"Complete MaleCNS · seed {seed}"),
            (f"pick-place-shuffled-seed{seed}-r0.mp4", f"Shuffled CNS #1 · seed {seed}"),
            (f"pick-place-shuffled-seed{seed}-r1.mp4", f"Shuffled CNS #2 · seed {seed}"),
        )
    )
    shots = "".join(
        media(site, MILESTONE / "screenshots" / name, "screenshots", caption)
        for name, caption in (
            (
                "ui-kitchen-chunk.png",
                "Kitchen live view with the motor-plan panel: the 10-step chunk decoded from "
                "the 68 front-leg motor neurons, one sparkline per joint.",
            ),
        )
    )
    notes_path = MILESTONE / "findings.json"
    notes = json.loads(notes_path.read_text()) if notes_path.is_file() else {}
    findings = "".join(
        f"<article class='finding'><span class='tag {esc(item['class'])}'>{esc(item['tag'])}</span>"
        f"<h3>{esc(item['title'])}</h3><p>{item['text']}</p></article>"
        for item in notes.get("findings", [])
    )
    status, pick_summary, topology = pick_place_tables()
    values = {
        "lede": notes.get("lede", "Results are filled in as runs complete."),
        "findings": findings,
        "caveats": "".join(f"<li>{item}</li>" for item in notes.get("caveats", [])),
        "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "kitchen_status": pill(kitchen),
        "kitchen_table": kitchen_table(kitchen),
        "kitchen_lesions": lesion_table(kitchen),
        "old_kitchen": kitchen_table(load(OLD_KITCHEN_RUN)),
        "sweep_table": sweep_table(),
        "kitchen_videos": kitchen_videos,
        "pick_status": status,
        "pick_summary": pick_summary,
        "topology": topology,
        "pick_videos": pick_videos,
        "shots": shots,
    }
    page = (Path(__file__).parent / "milestone_template.html").read_text()
    for key, value in values.items():
        page = page.replace("{{" + key + "}}", value)
    output = site / "index.html"
    output.write_text(page)
    return output


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else MILESTONE / "site"))
