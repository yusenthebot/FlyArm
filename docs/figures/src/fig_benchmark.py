"""Figure: the articulated multi-step manipulation benchmark.

(a) The scene: a Franka in front of a drawer cabinet, a lidded cabinet with a shelf, a bin and a
marked region, with scanned household objects; a real render of a reset. (b) The seven skills and
their physical success checks, articulations and object skills. (c) Task templates as sequences of
skills, three to eight subgoals; held-out compositions are dashed. (d) The five preregistered splits.

Facts: docs/MANIPULATION_ENV.md (skills, templates, splits), src/flyarm/manipulation/tasks.py and
splits.py. Raster input: assets/render-manipulation.png from render_tasks.py. Run from any directory:

    python fig_benchmark.py
    python qa_svg_figure.py ../benchmark-figure.svg --png --strict-tidy
"""

from pathlib import Path

from figkit import HAIR, INK, MUTED, PAL, Fig

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "benchmark-figure.svg"
ASSETS = HERE / "assets"

W, H = 1400, 460
f = Fig(W, H)
LABEL, MODULE, MIN = f.fs("label"), f.fs("module"), f.fs("min")
ART, OBJ = "purple", "blue"  # articulation skills, object skills

PY0, PY1 = 16, 452
WA, WB, WC, WD = 330, 230, 470, 290
XA, XB, XC, XD = f.place(20, 1380, (WA, WB, WC, WD))

# ================================================================ (a) scene
top = f.panel(XA, PY0, WA, PY1 - PY0, "gray", "(a) Scene", dashed=True)
IW = WA - 24
IH = round(IW * 538 / 717)
f.image(ASSETS / "render-manipulation.png", XA + 12, top + 4, IW, IH)
FUR = ("1 to 2 drawers", "lidded cabinet", "bin", "region")
cw = (IW - 10) / 2
for k, s_ in enumerate(FUR):
    x = XA + 12 + (k % 2) * (cw + 10)
    y = top + IH + 18 + (k // 2) * 46
    f.chip(round(x), round(y), round(cw), 36, s_, "gray", size=MIN, fill="#FFFFFF")

# ================================================================ (b) skills
top = f.panel(XB, PY0, WB, PY1 - PY0, "gray", "(b) Skills")
SKILLS = (("open drawer", ART), ("close drawer", ART), ("open lid", ART), ("close lid", ART),
          ("pick", OBJ), ("place", OBJ), ("stack", OBJ))
SH, SG = 40, 10
y0 = top + 4
for k, (s_, role) in enumerate(SKILLS):
    f.chip(XB + 12, y0 + k * (SH + SG), WB - 24, SH, s_, role, size=LABEL, fill=PAL[role].tint,
           color=PAL[role].deep)

# ================================================================ (c) templates
top = f.panel(XC, PY0, WC, PY1 - PY0, "gray", "(c) Task templates")
A_, O_ = ART, OBJ
TEMPLATES = (("put_away", (A_, O_, A_), False), ("retrieve", (A_, O_, O_, A_), False),
             ("tower", (O_, O_, O_), False), ("tidy", (A_, O_, A_, A_, O_, A_), False),
             ("unpack", (A_, O_, O_, A_, A_, O_, A_), False),
             ("full_cleanup", (A_, O_, A_, O_, O_, A_, O_, A_), True))
CELL, GAP = 30, 7
NAME_W = 150
RH = (PY1 - 12 - top) / len(TEMPLATES)
for r, (name, steps, held) in enumerate(TEMPLATES):
    cy = top + RH * r + RH / 2
    f.text(XC + 14, cy + 6, name, size=MIN, color=MUTED if held else INK, family="mono")
    for k, role in enumerate(steps):
        x = XC + 14 + NAME_W + k * (CELL + GAP)
        p = PAL[role]
        dash = ' stroke-dasharray="4 3"' if held else ""
        fill = "#FFFFFF" if held else p.tint
        f.add(f'<rect x="{x:.1f}" y="{cy - CELL / 2:.1f}" width="{CELL}" height="{CELL}" rx="4" fill="{fill}" '
              f'stroke="{p.accent}" stroke-width="1.2"{dash}/>')

# ================================================================ (d) splits
top = f.panel(XD, PY0, WD, PY1 - PY0, "gray", "(d) Splits")
SPLITS = (("train", "27 objects, 7 templates", False), ("iid test", "same distribution", False),
          ("unseen objects", "14 new objects", True), ("unseen furniture", "disjoint ranges", True),
          ("unseen composition", "4 new templates", True))
DH = (PY1 - 12 - top - 4 * 10) / len(SPLITS)
for k, (name, sub, held) in enumerate(SPLITS):
    y = top + k * (DH + 10)
    c = f.card(XD + 12, y, WD - 24, DH, fill="#FFFFFF", stroke=PAL["red"].accent if held else HAIR, dashed=held)
    f.text(XD + 26, y + DH / 2 - 4, name, size=LABEL, weight=500, box=c)
    f.text(XD + 26, y + DH / 2 + 20, sub, size=MIN, color=MUTED, box=c)

f.save(str(OUT))
print(OUT)
