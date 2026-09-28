"""Figure: the FlyArm pipeline - a frozen, complete fly CNS in closed loop with a simulated Franka.

One control step of the articulated manipulation benchmark, left to right: the 230-value state
observation (plus 70 fixed fine-scale copies) goes through the trained sensory encoder into the
1,846 ascending neurons of the frozen MaleCNS v1.0 connectome; three rate-dynamics steps later the
1,314 descending and 708 VNC motor neurons are read by the trained linear decoder into a 5-value
end-effector action, which inverse kinematics executes in MuJoCo; the next observation closes the
loop at 20 Hz. The strip below names the training stages, which touch only the encoder and decoder.

Facts come from src/flyarm (manipulation/sim.py observation layout, manipulation/features.py,
whole_brain/policy.py, whole_brain/backend_mlx.py, grasp/task.py) and docs/RESEARCH_LOG.md (E56 to
E61). Raster inputs: assets/rollout-0300.png from render_rollout.py (a real rollout of the reported
checkpoint) and assets/cns-soma-map.png from make_assets.py. Run from any directory:

    python fig_pipeline.py
    python qa_svg_figure.py ../pipeline-figure.svg --png --strict-tidy
"""

from pathlib import Path

from figkit import FAINT, HAIR, INK, MUTED, PAL, WIRE, Fig

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "pipeline-figure.svg"
ASSETS = HERE / "assets"
FIRE, SNOW = ASSETS / "fire.svg", ASSETS / "snowflake.svg"

W, H = 1400, 560
f = Fig(W, H)
LABEL, MODULE, MIN = f.fs("label"), f.fs("module"), f.fs("min")
MARK = 20  # trained / frozen mark size

# ---------------------------------------------------------------- grid
LANE_T = 20  # closed-loop lane above the panels
PY0, PY1 = 38, 452  # main panels
SY0, SY1 = 464, 552  # training strip
WA, WB, WC, WD, WE = 190, 200, 290, 372, 260
XA, XB, XC, XD, XE = f.place(20, 1380, (WA, WB, WC, WD, WE), gap=12)
Y = 268  # row line of the main flow
CARD_H = 292  # observation and encoder cards share one height, centred on the row line
CHIP_H = 36


def mark(icon, x, y):
    f.asset(icon, x, y, MARK, color=None)


def neurons(x, cy, w, count, name, role):
    """Neuron-group chip: count on top, group name below."""
    p = PAL[role]
    c = f.card(x, cy - 29, w, 58, fill=p.tint, stroke=p.accent)
    f.text(x + w / 2, cy - 4, count, size=MODULE, weight=500, color=p.deep, anchor="middle", box=c)
    f.text(x + w / 2, cy + 19, name, size=MIN, color=p.deep, anchor="middle", box=c)
    return c


# ================================================================ (a) observation
f.panel(XA, PY0, WA, PY1 - PY0, "gray", "(a) Input", dashed=True)
obs = f.card(XA + 10, Y - CARD_H / 2, WA - 20, CARD_H, fill="#FFFFFF", stroke=HAIR)
f.text(XA + 24, Y - CARD_H / 2 + 30, "Observation $o_t$", size=MODULE, weight=500, box=obs)
ROWS = (("arm state", "$q_t$", "gray", False), ("objects", "$x_t$", "gray", False),
        ("scene", "$s_t$", "gray", False), ("task cue", "$c_t$", "amber", True))
CW = WA - 44
CH2 = 44
cy = Y - CARD_H / 2 + 60
for name, sym, role, dashed in ROWS:
    c = f.card(XA + 22, cy, CW, CH2, fill="#FFFFFF", stroke=PAL[role].accent, dashed=dashed)
    f.text(XA + 34, cy + 28, name, size=LABEL, color=INK, box=c)
    f.text(XA + 22 + CW - 12, cy + 28, sym, size=LABEL, anchor="end", box=c)
    cy += CH2 + 12

# ================================================================ (b) sensory encoder
f.panel(XB, PY0, WB, PY1 - PY0, "amber", "(b) Encoder")
enc = f.card(XB + 10, Y - CARD_H / 2, WB - 20, CARD_H, "amber")
f.text(XB + 24, Y - CARD_H / 2 + 30, "Encoder $E_\\phi$", size=MODULE, weight=500, box=enc)
mark(FIRE, XB + WB - 24 - MARK, Y - CARD_H / 2 + 12)
f.mlp(XB + 30, Y - CARD_H / 2 + 56, WB - 60, 130, [4, 6, 6, 5], "amber")
asc = neurons(XB + 24, Y + CARD_H / 2 - 46, WB - 48, "1,846", "ascending", "amber")
f.connect(obs, enc)

# ================================================================ (c) frozen connectome
f.panel(XC, PY0, WC, PY1 - PY0, "gray", "(c) Fly CNS")
CX, CW2 = XC + 10, WC - 20
ctop = PY0 + 50
cns = f.card(CX, ctop, CW2, PY1 - 10 - ctop, fill="#FFFFFF", key=True)
f.text(CX + 14, ctop + 28, "MaleCNS v1.0", size=MODULE, weight=500, box=cns)
mark(SNOW, CX + CW2 - 14 - MARK, ctop + 10)
MAP_W = 150
MAP_H = round(MAP_W * 1215 / 900)
f.image(ASSETS / "cns-soma-map.png", CX + (CW2 - MAP_W) / 2, ctop + 40, MAP_W, MAP_H, fit="contain", frame=None, r=0)
ty = ctop + 40 + MAP_H + 20
f.text(CX + CW2 / 2, ty, "166,700 neurons", size=LABEL, anchor="middle", box=cns)
f.text(CX + CW2 / 2, ty + 24, "10.5M synapses", size=MIN, color=MUTED, anchor="middle", box=cns)
f.text(CX + CW2 / 2, ty + 54, "$h \\gets 0.5\\,h + 0.5\\tanh(I + 0.8\\,Wh)$", size=MIN,
       anchor="middle", box=cns)
f.text(CX + CW2 / 2, ty + 80, "$\\times 3$ per step", size=MIN, color=MUTED, anchor="middle", box=cns)
f.connect(asc, cns, sides=("right", "left"), tb=None)

# ================================================================ (d) motor readout
f.panel(XD, PY0, WD, PY1 - PY0, "blue", "(d) Motor readout")
NW = 108
N1 = XD + 12
dn = neurons(N1, Y - 36, NW, "1,314", "descending", "blue")
mn = neurons(N1, Y + 36, NW, "708", "VNC motor", "blue")
f.bus(cns, [dn, mn], side="right", ta=(Y - ctop) / (PY1 - 10 - ctop), at=XD - 6)
TW, TH = 60, 56
TX = N1 + NW + 36
wout = f.trapezoid(TX, Y - TH / 2, TW, TH, "red", "$W_{\\mathrm{out}}$", size=LABEL, direction="right")
mark(FIRE, TX + TW / 2 - MARK / 2, Y - TH / 2 - 8 - MARK)
MERGE = N1 + NW + 20
for src in (dn, mn):
    sx, sy = f.port(src, "right", out=1)
    f.line(f"M{sx:.1f} {sy:.1f}H{MERGE}", WIRE, 1.3)
f.line(f"M{MERGE} {Y - 36}V{Y + 36}", WIRE, 1.3)
f.dot(MERGE, Y, WIRE, 2.4)
f.arrow(f"M{MERGE} {Y}H{TX - 1}", WIRE, 1.3, head=6.5)
AX = TX + TW + 20
AW = XD + WD - 12 - AX
act = f.card(AX, Y - 46, AW, 92, fill="#FFFFFF", stroke=HAIR)
f.text(AX + AW / 2, Y - 16, "Action $a_t$", size=LABEL, anchor="middle", box=act)
CELL, GAP = 16, 8
vw = 5 * CELL + 4 * GAP
vx = AX + (AW - vw) / 2
f.vector(vx, Y + 6, 5, "blue", cell=CELL, vertical=False, gap=GAP, values=[0.55, 0.35, 0.75, 0.45, 0.9])
f.connect(wout, act)

# ================================================================ (e) environments
f.panel(XE, PY0, WE, PY1 - PY0, "gray", "(e) MuJoCo", dashed=True)
IW, IH = 176, 132  # two 4:3 renders stacked under the panel title
IX = XE + (WE - IW) / 2 + 8  # the bus trunk runs left of them
TASKS = (("rollout-0300.png", "manipulation"), ("render-kitchen.png", "FrankaKitchen"))
block = 2 * (IH + 30) + 16
iy = Y - block / 2
imgs = []
for name, label in TASKS:
    imgs.append(f.image(ASSETS / name, IX, iy, IW, IH))
    f.text(IX + IW / 2, iy + IH + 22, label, size=MIN, color=MUTED, anchor="middle", family="serif", italic=True)
    iy += IH + 30 + 16
f.bus(act, imgs, side="right", at=XE + 24)
img = imgs[0]

# ================================================================ closed loop: next observation
f.route(img, obs, (LANE_T,), sides=("top", "top"), ta=0.7, tb=0.8, color=FAINT, sw=1.3, dashed=True, head=7,
        label="next state", label_size=MIN, label_at=XC + WC / 2, knockout=True)

# ================================================================ (f) training
ps = f.panel(20, SY0, 1360, SY1 - SY0, "gray", "(f) Training", horizontal=True)
TY = SY0 + (SY1 - SY0 - 40) / 2
STAGES = (("Demonstrations", 190, "gray"), ("Imitation (DAgger)", 210, "red"),
          ("PPO + DAPG", 170, "red"))
xs = f.place(260, 1000, [w for _, w, _ in STAGES], gap=48)
chips = [f.chip(round(x), TY, w, 40, s_, role, size=LABEL, fill="#FFFFFF") for (s_, w, role), x in zip(STAGES, xs)]
for a, b in zip(chips, chips[1:]):
    f.connect(a, b)
LX = 1060
mark(FIRE, LX, TY + 10)
f.text(LX + MARK + 8, TY + 26, "trained $E_\\phi$, $W_{\\mathrm{out}}$", size=LABEL, box=ps)
mark(SNOW, LX + 200, TY + 10)
f.text(LX + 200 + MARK + 8, TY + 26, "frozen", size=LABEL, box=ps)

f.save(str(OUT))
print(OUT)
