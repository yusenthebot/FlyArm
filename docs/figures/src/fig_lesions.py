"""Figure: test-time lesions of the reported connectome controller (research log E63).

Full-task success on 112 iid test episodes (16 per template, the same episodes for every condition),
bars with Wilson 95% intervals and the number of successful episodes. (a) The whole network: the
recurrent state cleared before every control step, intact, and only the interface (input and output
neurons) left. (b) Each anatomical group silenced (purple, with a schematic of the fly CNS marking the
group) between its two random sets of the same size drawn from all non-interface neurons (gray). The
dashed line is the intact level. Brackets give exact two-sided McNemar p on the same episodes: in (a)
against intact, in (b) of the anatomical lesion against each of its random sets (computed here from the
per-episode outcomes, as flyarm.whole_brain.lesion.mcnemar does).

Data: docs/results/manipulation-lesions.json (scripts/lesion_analysis.py). The English version is the
paper figure; `--lang zh` writes the Chinese version for the project website. Run from any directory:

    python fig_lesions.py
    python fig_lesions.py --lang zh
    python qa_svg_figure.py ../lesion-figure.svg --png --strict-tidy --words-per-10k 2.3 --min-coverage 0.25
    python qa_svg_figure.py ../../../site/figures/lesion-figure-zh.svg --png --strict-tidy --words-per-10k 2.3 --min-coverage 0.25
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from figkit import FAINT, INK, MUTED, PAL, Fig, text_w

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "docs" / "results" / "manipulation-lesions.json"
OUTS = {"en": ROOT / "docs" / "figures" / "lesion-figure.svg", "zh": ROOT / "site" / "figures" / "lesion-figure-zh.svg"}

GROUPS = ("vnc_interneurons", "optic_lobes", "central_brain", "sensory")
TEXT = {
    "en": {
        "panel_a": "(a) Whole network", "panel_b": "(b) Silenced groups",
        "state_reset": "State reset every step", "intact": "Intact", "silence_all_but_interface": "Interface only",
        "vnc_interneurons": "VNC interneurons", "optic_lobes": "Optic lobes", "central_brain": "Central brain",
        "sensory": "Sensory neurons", "neurons": "{n} neurons",
        "region": "anatomical group", "random": "random, same size", "of": "of {n}",
        "axis": "Full-task success (%)",
    },
    "zh": {
        "panel_a": "(a) 整个网络", "panel_b": "(b) 按解剖分组关掉",
        "state_reset": "每一步清空记忆", "intact": "完整的大脑", "silence_all_but_interface": "只留输入输出神经元",
        "vnc_interneurons": "VNC 中间神经元", "optic_lobes": "视叶", "central_brain": "中央脑",
        "sensory": "感觉神经元", "neurons": "{n} 个神经元",
        "region": "解剖分组", "random": "随机同样数量", "of": "共 {n} 次",
        "axis": "完整任务成功率 (%)",
    },
}
COLOR = {"intact": "green", "state_reset": "red", "silence_all_but_interface": "purple", "region": "purple",
         "random": "gray"}

# ---------------------------------------------------------------- statistics


def mcnemar_p(a: list[int], b: list[int]) -> float:
    """Exact two-sided McNemar p on paired outcomes (binomial test on the discordant pairs)."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must cover the same episodes")
    lost = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    gained = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def p_label(p: float) -> str:
    if p >= 0.001:
        return f"$p = {p:.2g}$"
    mant, exp = f"{p:.0e}".split("e")
    return f"$p = {mant}\\times10^{{{int(exp)}}}$"


# ---------------------------------------------------------------- drawing

W = 700
f = Fig(W, 1)  # height set from the layout below
LABEL, MIN, TITLE = f.fs("label"), f.fs("min"), f.fs("title")

PANEL_X0, PANEL_X1 = 8, 692
GLYPH_CX = 42                    # centre of the CNS schematics
LABEL_R = 232                    # right edge of the row labels
PX0, PX1 = 240, 520              # 0 % and 100 % of the success axis
COUNT_R = 552                    # right edge of the success counts
BRACKET_X = 567                  # vertical stroke of the significance brackets
PITCH, BAR_H, CAP = 26, 16, 8    # row pitch, bar height, error-bar cap height
BLOCK = 3 * PITCH                # one block: comparison, reference, comparison
GROUP_GAP = 14
HEAD = round(TITLE * 1.47) + round(TITLE * 0.8)  # panel title row, as figkit.panel lays it out
PAD, GUTTER, AXIS = 10, 8, 60                   # panel bottom padding, gap between panels, axis band
HA = HEAD + BLOCK + PAD
HB = HEAD + len(GROUPS) * BLOCK + (len(GROUPS) - 1) * GROUP_GAP + PAD
f.h = H = 8 + HA + GUTTER + HB + AXIS


def x_of(rate: float) -> float:
    return PX0 + rate * (PX1 - PX0)


def bar(cy: float, row: dict, role: str) -> None:
    p = PAL[role]
    rate = row["successes"] / row["episodes"]
    fill = p.accent if role != "gray" else p.mid
    if rate > 0:
        f.card(PX0, round(cy - BAR_H / 2, 1), round(x_of(rate) - PX0, 1), BAR_H, fill=fill, stroke=p.deep if role != "gray" else p.accent,
               sw=0.8, r=1.5)
    lo, hi = (x_of(v) for v in row["wilson_95"])
    f.line(f"M{lo:.1f} {cy}H{hi:.1f}", INK, 1.1)
    for x in (lo, hi):
        f.line(f"M{x:.1f} {cy - CAP / 2}V{cy + CAP / 2}", INK, 1.1)
    f.text(COUNT_R, cy + MIN * 0.36, str(row["successes"]), size=MIN, anchor="end")


def bracket(y1: float, y2: float, p: float) -> None:
    """Bracket from the row at y1 to the row at y2 with its p value on the right."""
    f.line(f"M{BRACKET_X - 7} {y1:.1f}H{BRACKET_X}V{y2:.1f}H{BRACKET_X - 7}", INK if p < 0.05 else FAINT, 1.0)
    f.text(BRACKET_X + 7, (y1 + y2) / 2 + MIN * 0.36, p_label(p), size=MIN, color=INK if p < 0.05 else MUTED)


def grid(y0: float, y1: float) -> None:
    for k in range(1, 5):
        f.line(f"M{x_of(k / 4):.1f} {y0}V{y1}", "#FFFFFF", 1.2)
    f.line(f"M{PX0} {y0}V{y1}", FAINT, 1.0)


def cns(cx: float, top: float, part: str | None) -> None:
    """Schematic fly CNS (brain with optic lobes above the neck connective and the VNC), 70 px tall,
    with `part` filled in the anatomical-group colour."""
    on, off = PAL["purple"], PAL["gray"]

    def style(name: str) -> str:
        p = on if name == part else off
        return f'fill="{p.accent if name == part else p.tint}" stroke="{p.deep if name == part else p.accent}" stroke-width="0.9"'

    t = top
    # optic lobes and central brain
    f.add(f'<ellipse cx="{cx - 18}" cy="{t + 12}" rx="9" ry="10" {style("optic_lobes")}/>')
    f.add(f'<ellipse cx="{cx + 18}" cy="{t + 12}" rx="9" ry="10" {style("optic_lobes")}/>')
    f.add(f'<ellipse cx="{cx}" cy="{t + 12}" rx="8.5" ry="11" {style("central_brain")}/>')
    # neck connective
    f.add(f'<rect x="{cx - 2.5}" y="{t + 23}" width="5" height="7" {style("neck")}/>')
    # VNC: three thoracic neuromeres and the abdominal tip
    d = (f"M{cx} {t + 30}C{cx + 7} {t + 30} {cx + 9} {t + 34} {cx + 8} {t + 38}C{cx + 9} {t + 42} {cx + 9} {t + 46} "
         f"{cx + 7} {t + 48}C{cx + 9} {t + 51} {cx + 9} {t + 56} {cx + 6} {t + 59}C{cx + 5} {t + 63} {cx + 2} {t + 68} "
         f"{cx} {t + 70}C{cx - 2} {t + 68} {cx - 5} {t + 63} {cx - 6} {t + 59}C{cx - 9} {t + 56} {cx - 9} {t + 51} "
         f"{cx - 7} {t + 48}C{cx - 9} {t + 46} {cx - 9} {t + 42} {cx - 8} {t + 38}C{cx - 9} {t + 34} {cx - 7} {t + 30} {cx} {t + 30}Z")
    f.add(f'<path d="{d}" {style("vnc_interneurons")}/>')
    # sensory afferents: leg nerves entering the VNC and antennal nerves entering the brain
    sens = on.accent if part == "sensory" else off.accent
    sw = 1.8 if part == "sensory" else 1.0
    nerves = [(cx + s * 8.5, t + y, cx + s * 17, t + y + 4) for s in (-1, 1) for y in (37, 47, 56)]
    nerves += [(cx + s * 4, t + 2, cx + s * 8, t - 4) for s in (-1, 1)]
    for x1, y1, x2, y2 in nerves:
        f.add(f'<path d="M{x1:.1f} {y1:.1f}L{x2:.1f} {y2:.1f}" fill="none" stroke="{sens}" stroke-width="{sw}" '
              f'stroke-linecap="round"/>')


def build(lang: str) -> Path:
    s = TEXT[lang]
    data = json.loads(DATA.read_text(encoding="utf-8"))["conditions"]
    intact = data["intact"]

    # (a) the whole network: comparison, reference, comparison
    ya = 8
    hb = HA
    top = f.panel(PANEL_X0, ya, PANEL_X1 - PANEL_X0, hb, "gray", s["panel_a"])
    rows_a = ("state_reset", "intact", "silence_all_but_interface")
    a1 = top + BLOCK
    f.text(COUNT_R, ya + round(TITLE * 1.47), s["of"].format(n=intact["episodes"]), size=MIN, color=MUTED,
           anchor="end")
    grid(top, a1)
    cys = [top + PITCH * (k + 0.5) for k in range(3)]
    for cy, key in zip(cys, rows_a, strict=True):
        f.text(LABEL_R, cy + LABEL * 0.36, s[key], size=LABEL, weight=500, anchor="end",
               color=PAL[COLOR[key]].deep)
        bar(cy, data[key], COLOR[key])
    bracket(cys[0], cys[1] - 2, data["state_reset"]["mcnemar"]["p"])
    bracket(cys[1] + 2, cys[2], data["silence_all_but_interface"]["mcnemar"]["p"])
    ref = [(top, a1)]

    # (b) anatomical groups between their random sets
    yb = ya + hb + GUTTER
    yb1 = yb + HB
    top_b = f.panel(PANEL_X0, yb, PANEL_X1 - PANEL_X0, yb1 - yb, "gray", s["panel_b"])
    # legend in the title row
    lx = PANEL_X1 - 12
    for role, key in (("gray", "random"), ("purple", "region")):
        p = PAL[role]
        tw = text_w(s[key], MIN)
        f.text(lx, yb + round(TITLE * 1.47), s[key], size=MIN, color=MUTED, anchor="end")
        sx = lx - tw - 8 - 14
        f.add(f'<rect x="{sx:.1f}" y="{yb + round(TITLE * 1.47) - 12}" width="14" height="12" rx="1.5" '
              f'fill="{p.accent if role == "purple" else p.mid}" stroke="{p.deep if role == "purple" else p.accent}" stroke-width="0.8"/>')
        lx = sx - 20
    content_bottom = top_b + len(GROUPS) * BLOCK + (len(GROUPS) - 1) * GROUP_GAP
    grid(top_b, content_bottom)
    ref.append((top_b, content_bottom))
    for g, key in enumerate(GROUPS):
        region = data[f"silence_{key}"]
        randoms = [data[f"random_as_{key}_{d}"] for d in (0, 1)]
        b0 = top_b + g * (BLOCK + GROUP_GAP)
        cys = [b0 + PITCH * (k + 0.5) for k in range(3)]
        cns(GLYPH_CX, b0 + 4, key)
        f.text(LABEL_R, cys[1] - 1, s[key], size=LABEL, weight=500, anchor="end", color=PAL["purple"].deep)
        f.text(LABEL_R, cys[1] + 19, s["neurons"].format(n=f"{region['silenced_neurons']:,}"), size=MIN,
               color=MUTED, anchor="end")
        bar(cys[0], randoms[0], "gray")
        bar(cys[1], region, "purple")
        bar(cys[2], randoms[1], "gray")
        bracket(cys[0], cys[1] - 2, mcnemar_p(region["outcomes"], randoms[0]["outcomes"]))
        bracket(cys[1] + 2, cys[2], mcnemar_p(region["outcomes"], randoms[1]["outcomes"]))

    # intact reference line through both panels
    xi = x_of(intact["successes"] / intact["episodes"])
    for y0, y1 in ref:
        f.line(f"M{xi:.1f} {y0}V{y1}", PAL["green"].accent, 1.2, dashed=True)

    # shared x axis under panel (b)
    ay = yb1 + 6
    for k in range(5):
        x = x_of(k / 4)
        f.line(f"M{x:.1f} {ay}V{ay + 5}", FAINT, 1.0)
        f.text(x, ay + 22, str(25 * k), size=MIN, color=MUTED, anchor="middle")
    f.text((PX0 + PX1) / 2, ay + 46, s["axis"], size=MIN, anchor="middle")

    out = OUTS[lang]
    out.parent.mkdir(parents=True, exist_ok=True)
    f.save(str(out))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lang", choices=sorted(TEXT), default="en")
    print(build(parser.parse_args().lang))


if __name__ == "__main__":
    main()
