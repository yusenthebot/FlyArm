"""Supplementary figure: FlyArm infrastructure - data sources, the build step, the run directory, monitoring and
the research record.

(a) Upstream data, each under its own license: the MaleCNS v1.0 connectome, the MuJoCo Menagerie Franka Panda,
    the Gymnasium-Robotics FrankaKitchen and Google Scanned Objects.
(b) The connectome is compiled into a pack of edges with at least three synaptic contacts; scenes and objects are
    downloaded at pinned commits and hashes and are not committed.
(c) Every training or evaluation run writes one directory: its config, curves, evaluations, checkpoints and the
    latest progress clip.
(d) The dashboard (FastAPI, port 8780) reads the run directories: curves, evaluations, logs and rollouts; the live
    progress clip keeps only the newest segment of each run.
(e) The research log (entries E1 to E55 with a claims ledger), the result files and the milestone documents are
    in git.
Screenshot and clip frame are real: `flyarm dashboard` with a kitchen PPO run selected, and a frame of that run's
progress/latest.mp4; see assets/ASSETS.md.
"""

from __future__ import annotations

from pathlib import Path

from figkit import HAIR, INK, MUTED, PAL, WIRE, Fig, text_w

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "infrastructure-figure.svg"
A = HERE / "assets"

W, H = 1400, 454
f = Fig(W, H)
TITLE, LABEL, SMALL = f.fs("title"), f.fs("label"), f.fs("min")
GRY = PAL["gray"]


def link(a, b, sides=("right", "left"), **kw):
    kw.setdefault("color", WIRE)
    kw.setdefault("sw", 1.3)
    kw.setdefault("head", 7)
    return f.connect(a, b, sides=sides, **kw)


def lines(x, y, items, box, color=INK, pitch=22, family="sans", italic=False):
    for i, s in enumerate(items):
        f.text(x, y + pitch * i, s, size=SMALL, color=color, box=box, family=family, italic=italic)


# ================================================================ grid
LEFT = f.place(8, 794, (310, 210, 250), gap=8)  # (a), (b), (c) run the full height
PX = (LEFT[0], LEFT[1], LEFT[2])
PW = (310, 210, 250)
RX, RW = 802, 590  # (d) above (e)
TOP, BOTTOM = 8, H - 8
D_Y, D_H = TOP, 280
E_Y, E_H = D_Y + D_H + 8, BOTTOM - (D_Y + D_H + 8)
CT = 64  # content top under the panel titles
ROW_H, ROW_GAP = 72, 20
ROWS = [CT + k * (ROW_H + ROW_GAP) for k in range(4)]  # the four source rows

f.panel(PX[0], TOP, PW[0], BOTTOM - TOP, "gray", "(a) Sources", title_size=TITLE)
f.panel(PX[1], TOP, PW[1], BOTTOM - TOP, "gray", "(b) Build", title_size=TITLE)
f.panel(PX[2], TOP, PW[2], BOTTOM - TOP, "gray", "(c) Run", title_size=TITLE)
f.panel(RX, D_Y, RW, D_H, "gray", "(d) Monitor", title_size=TITLE)
f.panel(RX, E_Y, RW, E_H, "gray", "(e) Record", sub="in git", title_size=TITLE)

# ================================================================ (a) sources
SOURCES = (
    ("MaleCNS v1.0", "connectome", "CC BY 4.0"),
    ("MuJoCo Menagerie", "Franka Panda", "Apache-2.0"),
    ("Gymnasium-Robotics", "FrankaKitchen", "Apache-2.0"),
    ("Scanned Objects", "Google", "CC BY 4.0"),
)
AX, AW = PX[0] + 16, PW[0] - 32
sources = []
for (name, what, license_), y in zip(SOURCES, ROWS, strict=True):
    card = f.card(AX, y, AW, ROW_H, fill="#FFFFFF", stroke=HAIR)
    f.text(AX + 12, y + 26, name, size=SMALL, weight=600, color=INK, box=card)
    f.text(AX + 12, y + 54, what, size=SMALL, color=MUTED, box=card, family="serif", italic=True)
    lw = text_w(license_, SMALL) + 16  # on the second line, so a long name keeps the full width
    f.chip(AX + AW - 12 - lw, y + 40, lw, 26, license_, size=SMALL, fill=GRY.tint, stroke=GRY.mid,
           color=INK)
    sources.append(card)

# ================================================================ (b) build
BX, BW = PX[1] + 16, PW[1] - 32
pack = f.card(BX, ROWS[0], BW, ROW_H, "amber")
f.text(BX + 12, ROWS[0] + 30, "connectome pack", size=SMALL, weight=600, color=PAL["amber"].deep, box=pack)
f.text(BX + 12, ROWS[0] + 54, "$\\ge 3$ contacts", size=SMALL, color=INK, box=pack)
ASSET_Y = ROWS[1]
ASSET_H = ROWS[3] + ROW_H - ROWS[1]
scenes = f.card(BX, ASSET_Y, BW, ASSET_H, fill="#FFFFFF", stroke=HAIR)
f.text(BX + 12, ASSET_Y + 30, "scenes, objects", size=SMALL, weight=600, color=INK, box=scenes)
lines(BX + 12, ASSET_Y + 58, ("pinned commits", "SHA-256 hashes", "not committed"), scenes, color=MUTED, pitch=24,
      family="serif", italic=True)
for k, (what, pin) in enumerate((("Panda", "822c2d8"), ("kitchen", "1.4.2"), ("objects", "6ff8d27"))):
    y = ASSET_Y + 150 + 24 * k  # in the order of the sources on the left
    f.text(BX + 12, y, what, size=SMALL, color=MUTED, box=scenes, family="serif", italic=True)
    f.text(BX + 82, y, pin, size=SMALL, color=INK, box=scenes, family="mono")
link(sources[0], pack)
for src, y in zip(sources[1:], ROWS[1:], strict=True):
    link(src, scenes, ta=0.5, tb=(y + ROW_H / 2 - ASSET_Y) / ASSET_H)

# ================================================================ (c) run directory
CX, CW = PX[2] + 16, PW[2] - 32
RUN_H = ROWS[3] + ROW_H - ROWS[0]
run = f.card(CX, ROWS[0], CW, RUN_H, fill="#FFFFFF", key=True)
f.text(CX + 12, ROWS[0] + 30, "runs/<run>/", size=SMALL, weight=700, color=INK, box=run, family="mono")
f.text(CX + 12, ROWS[0] + 56, "training, evaluation", size=SMALL, color=MUTED, box=run, family="serif",
       italic=True)
lines(CX + 16, ROWS[0] + 96, ("config.json", "results.json", "curves.json", "evaluations.json",
                              "policy-*.safetensors", "progress/latest.mp4"), run, family="mono", pitch=30)
CHIP_Y = ROWS[3] + ROW_H - 12 - 30
for k, s in enumerate(("MLX", "mjbatch")):
    f.chip(CX + 12 + k * 96, CHIP_Y, 86, 30, s, size=SMALL, fill=GRY.tint, stroke=GRY.accent, color=INK,
           family="mono")
link(pack, run, ta=0.5, tb=(ROWS[0] + ROW_H / 2 - ROWS[0]) / RUN_H)
link(scenes, run, ta=0.5, tb=(ASSET_Y + ASSET_H / 2 - ROWS[0]) / RUN_H)

# ================================================================ (d) monitor: real screenshot and clip frame
IMG_H = 180
DASH_W, CLIP_W = 320, 180
DX = f.place(RX + 16, RX + RW - 16, (DASH_W, CLIP_W), gap=24)
IMG_Y = D_Y + 58
dash = f.image(A / "dashboard.png", DX[0], IMG_Y, DASH_W, IMG_H, fit="cover")
clip = f.image(A / "progress-clip.png", DX[1], IMG_Y, CLIP_W, IMG_H, fit="cover")
CAP_Y = IMG_Y + IMG_H + 24
f.text(DX[0] + DASH_W / 2, CAP_Y, "dashboard, FastAPI :8780", size=SMALL, anchor="middle", color=MUTED)
f.text(DX[1] + CLIP_W / 2, CAP_Y, "Live: latest clip", size=SMALL, anchor="middle", color=MUTED)
link(dash, clip)
link(run, dash, ta=(IMG_Y + IMG_H / 2 - ROWS[0]) / RUN_H, tb=0.5)

# ================================================================ (e) record
REC_Y = E_Y + 58
REC_H = E_H - 58 - 14
GIT = (RX + 24, REC_Y + REC_H / 2 - 18, 36)
f.asset(A / "git.svg", *GIT, INK, sw=2)
git = f.zone(GIT[0], GIT[1], GIT[2], GIT[2])  # the wire lands on the icon's own box
RECORDS = (("RESEARCH_LOG.md", "E1-E55, claims"), ("results/", "35 result files"),
           ("MILESTONE_KITCHEN.md", "milestone"))
# each column is as wide as its longer line (monospace is 0.6 em per character); the gaps take the rest
R_W = [max(len(path) * 0.6 * SMALL, text_w(what, SMALL)) for path, what in RECORDS]
R_X = f.place(GIT[0] + GIT[2] + 16, RX + RW - 16, R_W)
assert min(R_X[i + 1] - R_X[i] - R_W[i] for i in range(len(R_X) - 1)) >= 20, "record columns too tight"
for (path, what), x in zip(RECORDS, R_X, strict=True):
    f.text(x, REC_Y + REC_H / 2 - 4, path, size=SMALL, color=INK, family="mono")
    f.text(x, REC_Y + REC_H / 2 + 20, what, size=SMALL, color=MUTED, family="serif", italic=True)
link(run, git, ta=(REC_Y + REC_H / 2 - ROWS[0]) / RUN_H, tb=0.5)

f.save(str(OUT))
print(OUT)
