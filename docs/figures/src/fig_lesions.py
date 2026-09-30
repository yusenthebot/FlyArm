"""Figure: test-time lesions of the reported connectome controller (research log E63).

Full-task success on 112 iid test episodes (16 per template, the same episodes for every
condition) with Wilson 95% intervals: intact, the state reset before every control step, each
anatomical group silenced, and two random sets of the same size drawn from all non-interface
neurons. Data: docs/results/manipulation-lesions.json (scripts/lesion_analysis.py). Run from any
directory:

    python fig_lesions.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "results" / "manipulation-lesions.json"
OUT = HERE.parent / "lesion-figure"

INK, MUTED, FAINT, HAIR = "#1F1F1F", "#5C5C5C", "#8C8C8C", "#D8D4CA"
REGION, RANDOM, INTACT, RESET = "#8870B8", "#9C968A", "#6F9E57", "#C4583C"
GROUPS = (
    ("central_brain", "Central brain"),
    ("optic_lobes", "Optic lobes"),
    ("vnc_interneurons", "VNC interneurons"),
    ("sensory", "Sensory neurons"),
)


def rows(conditions: dict) -> list[tuple[str, str, str]]:
    """(condition key, label, color) from top to bottom."""
    out = [("intact", "Intact", INTACT), ("state_reset", "State reset every step", RESET)]
    for key, label in GROUPS:
        if f"silence_{key}" not in conditions:
            continue
        count = conditions[f"silence_{key}"]["silenced_neurons"]
        out.append((f"silence_{key}", f"{label} ({count:,})", REGION))
        out += [(f"random_as_{key}_{d}", f"random, same size ({d + 1})", RANDOM) for d in (0, 1)]
    out.append(("silence_all_but_interface", "All but the interface", REGION))
    return [row for row in out if row[0] in conditions]


def main() -> None:
    conditions = json.loads(DATA.read_text())["conditions"]
    table = rows(conditions)
    plt.rcParams.update(
        {"font.family": ["Helvetica Neue", "Helvetica", "Arial"], "font.size": 6.7, "svg.fonttype": "none"}
    )
    fig, ax = plt.subplots(figsize=(3.5, 0.17 * len(table) + 0.45))
    for y, (key, _, color) in enumerate(table):
        row = conditions[key]
        rate = 100 * row["success_rate"]
        low, high = (100 * v for v in row["wilson_95"])
        ax.plot([low, high], [y, y], color=color, lw=1.2, solid_capstyle="butt")
        ax.plot(rate, y, "o", color=color, ms=3.6, mec="white", mew=0.5)
        text = f"{row['successes']}/{row['episodes']}"
        if "mcnemar" in row:
            p = row["mcnemar"]["p"]
            text += f"  p={p:.1g}" if p < 0.001 else f"  p={p:.2f}"
        ax.text(103, y, text, va="center", ha="left", color=MUTED, fontsize=6.0)
    intact = 100 * conditions["intact"]["success_rate"]
    ax.axvline(intact, color=HAIR, lw=0.8, zorder=0)
    ax.set_yticks(range(len(table)), [label for _, label, _ in table])
    for tick, (_, _, color) in zip(ax.get_yticklabels(), table, strict=True):
        tick.set_color(INK if color != RANDOM else FAINT)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Full-task success, iid test (%)", color=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(FAINT)
    ax.tick_params(colors=MUTED, length=2, width=0.6)
    fig.subplots_adjust(left=0.36, right=0.8, top=0.97, bottom=0.2)
    for suffix in (".svg", ".png"):
        fig.savefig(OUT.with_suffix(suffix), dpi=300)
    print(OUT.with_suffix(".png"))


if __name__ == "__main__":
    main()
