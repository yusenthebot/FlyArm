"""Figure: FlyArm control network - a frozen, complete fly CNS in closed loop with a simulated Franka.

Two interface instantiations of one frozen MaleCNS v1.0 connectome:
row 1 is B1a pick and place (ascending neurons in, descending + VNC motor neurons out, end-effector
action), row 2 is B2 FrankaKitchen with the arm as the fly's left front leg (leg proprioceptors and
head sensory neurons in, leg motor neurons out, joint velocities through an action chunk and ACT's
temporal ensemble). Only the linear encoders W_in and decoders W_out are trained.

Facts come from src/flyarm (whole_brain/backend_mlx.py, whole_brain/policy.py, whole_brain/interface.py,
flyleg/interface.py, config.py) and docs/WHOLE_BRAIN.md, docs/B2_FLYLEG_GOAL.md. Raster inputs are built
by make_assets.py from the project's own data. Run from any directory:

    python fig_control_network.py
    python qa_svg_figure.py ../control_network-figure.svg --png --strict-tidy
"""

from pathlib import Path

from figkit import FAINT, HAIR, INK, MUTED, PAL, WIRE, Fig, text_w

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "control_network-figure.svg"
ASSETS = HERE / "assets"

W, H = 1400, 644
f = Fig(W, H)
AMB, BLU, RED, GRY = (PAL[k] for k in ("amber", "blue", "red", "gray"))
LABEL, MODULE, MIN, TITLE = f.fs("label"), f.fs("module"), f.fs("min"), f.fs("title")
FIRE, SNOW = ASSETS / "fire.svg", ASSETS / "snowflake.svg"
MARK = 20  # trained / frozen mark size

# ---------------------------------------------------------------- grid
LANE_L, LANE_T, LANE_B = 16, 16, 572  # closed-loop lanes: left edge, top, bottom
PY0, PY1 = 32, 556  # main panels
SY0, SY1 = 588, 636  # controls strip
WA, WB, WC = 484, 314, 518
XA, XB, XC = f.place(28, 1368, (WA, WB, WC), gap=12)
XR = XC + WC + 12  # right lane
Y1, Y2 = 178, 412  # row lines: B1a pick and place, B2 kitchen
DY = 44  # half pitch of a two-channel row
IMG_W, IMG_H = 186, 138
TRAP_W, TRAP_H = 60, 52
CHIP_W, CHIP_H = 140, 58
OCHIP_W = 124


def neurons(x, cy, count, name, role, w):
    """Neuron-group chip: count on top, group name below."""
    p = PAL[role]
    c = f.card(x, cy - CHIP_H / 2, w, CHIP_H, fill=p.tint, stroke=p.accent)
    f.text(x + w / 2, cy - 4, count, size=MODULE, weight=500, color=p.deep, anchor="middle", box=c)
    f.text(x + w / 2, cy + 19, name, size=MIN, color=p.deep, anchor="middle", box=c)
    return c


def linear(x, cy, sym, narrow, mark_side="top"):
    """Trainable linear map drawn as a trapezoid, with the trained mark above or below it."""
    t = f.trapezoid(x, cy - TRAP_H / 2, TRAP_W, TRAP_H, "red", sym, size=LABEL, direction=narrow)
    my = cy - TRAP_H / 2 - 6 - MARK if mark_side == "top" else cy + TRAP_H / 2 + 6
    f.asset(FIRE, x + TRAP_W / 2 - MARK / 2, my, MARK, color=None)
    return t


def row_label(cx, y, tag, name, box):
    f.add(f'<text x="{cx:.1f}" y="{y:.1f}" font-size="{LABEL}" text-anchor="middle" fill="{INK}" data-in="{box}">'
          f'<tspan font-weight="700">{tag}</tspan><tspan> {name}</tspan></text>')


# ================================================================ (a) sensory interface
pa = f.panel(XA, PY0, WA, PY1 - PY0, "amber", "(a) Sensory interface")
PA_BOX = f"panel{f._n}"
IMG_X = XA + 12
CHIP_X = XA + WA - 12 - CHIP_W
TRAP_X = IMG_X + IMG_W + 44

img1 = f.image(ASSETS / "render-pick-place.png", IMG_X, Y1 - IMG_H / 2, IMG_W, IMG_H)
row_label(IMG_X + IMG_W / 2, Y1 + IMG_H / 2 + 24, "B1a", "pick-and-place", PA_BOX)
win1 = linear(TRAP_X, Y1, "$W_{\\mathrm{in}}$", "left")
f.connect(img1, win1, label="$s_t$", label_size=LABEL)
in1 = neurons(CHIP_X, Y1, "1,846", "ascending", "amber", CHIP_W)
f.connect(win1, in1)

img2 = f.image(ASSETS / "render-kitchen.png", IMG_X, Y2 - IMG_H / 2, IMG_W, IMG_H)
row_label(IMG_X + IMG_W / 2, Y2 + IMG_H / 2 + 24, "B2", "FrankaKitchen", PA_BOX)
f.text(IMG_X + IMG_W / 2, Y2 + IMG_H / 2 + 46, "arm = left front leg", size=MIN, color=MUTED, anchor="middle",
       box=PA_BOX, family="serif", italic=True)
win2q = linear(TRAP_X, Y2 - DY, "$W_{\\mathrm{in}}$", "left", "top")
win2o = linear(TRAP_X, Y2 + DY, "$W_{\\mathrm{in}}$", "left", "bottom")
trunk = f.bus(img2, [win2q, win2o], side="right", at=IMG_X + IMG_W + 14)
for t, sym in ((win2q, "$q_t$"), (win2o, "$o_t$")):
    ty = f.port(t, "left")[1]
    f.text((trunk + TRAP_X) / 2, ty - 8, sym, size=LABEL, anchor="middle", family="serif")
in2q = neurons(CHIP_X, Y2 - DY, "23", "proprioceptors", "amber", CHIP_W)
in2o = neurons(CHIP_X, Y2 + DY, "4,868", "head sensory", "amber", CHIP_W)
f.connect(win2q, in2q)
f.connect(win2o, in2o)

# ================================================================ (b) frozen connectome
pb = f.panel(XB, PY0, WB, PY1 - PY0, "gray", "(b) Frozen connectome")
CX, CW = XB + 10, WB - 20
cns = f.card(CX, pb, CW, PY1 - 10 - pb, fill="#FFFFFF", key=True)
f.text(CX + 14, pb + 28, "MaleCNS v1.0", size=MODULE, weight=500, box=cns)
f.asset(SNOW, CX + CW - 14 - MARK, pb + 12, MARK, color=None)
MAP_W = 232
MAP_H = round(MAP_W * 1215 / 900)
MAP_X, MAP_Y = CX + (CW - MAP_W) / 2, pb + 42
f.image(ASSETS / "cns-soma-map.png", MAP_X, MAP_Y, MAP_W, MAP_H, fit="contain", frame=None, r=0)
LEG_Y = MAP_Y + MAP_H - 30
for k, (role, s) in enumerate((("amber", "in"), ("blue", "out"))):
    lx = CX + 18
    ly = LEG_Y + 22 * k
    f.dot(lx + 5, ly - 5, PAL[role].accent, 5)
    f.text(lx + 16, ly, s, size=MIN, color=MUTED, box=cns)
cy0 = MAP_Y + MAP_H + 26
f.text(CX + CW / 2, cy0, "166,700 neurons · 10.5M edges", size=MIN, color=MUTED, anchor="middle", box=cns)
f.text(CX + CW / 2, cy0 + 30, "$h \\gets 0.5h + 0.5\\,\\tanh(I + 0.8Wh)$", size=LABEL, anchor="middle", box=cns)
f.text(CX + CW / 2, cy0 + 58, "$\\times 3$ per step · 4 ms", size=MIN, color=MUTED, anchor="middle", box=cns)

# ================================================================ (c) motor interface
pc = f.panel(XC, PY0, WC, PY1 - PY0, "blue", "(c) Motor interface")
PC_BOX = f"panel{f._n}"
C1_W, C4_W, C5_W = OCHIP_W, 112, 104
C1 = XC + 12
C3 = C1 + C1_W + 54
C4 = C3 + TRAP_W + 20
C5 = C4 + C4_W + 20

# legend in the title row
lg_x = XC + WC - 12
for icon, word in ((SNOW, "frozen"), (FIRE, "trained")):
    tw = len(word) * MIN * 0.52
    f.text(lg_x, PY0 + 35, word, size=MIN, color=MUTED, anchor="end")
    f.asset(icon, lg_x - tw - 6 - 18, PY0 + 35 - 15, 18, color=None)
    lg_x -= tw + 6 + 18 + 18

# row 1: descending + VNC motor neurons -> W_out -> (dx, dy, dz, g) -> IK
dn = neurons(C1, Y1 - 37, "1,314", "descending", "blue", C1_W)
mn = neurons(C1, Y1 + 37, "708", "VNC motor", "blue", C1_W)
f.bus(cns, [dn, mn], side="right", ta=(Y1 - f.rect(cns)[1]) / f.rect(cns)[3], at=XC - 6)
wout1 = linear(C3, Y1, "$W_{\\mathrm{out}}$", "right")
MERGE = C1 + C1_W + 24
for src in (dn, mn):
    sx, sy = f.port(src, "right", out=1)
    f.line(f"M{sx:.1f} {sy:.1f}H{MERGE}", WIRE, 1.3)
f.line(f"M{MERGE} {Y1 - 37}V{Y1 + 37}", WIRE, 1.3)
f.dot(MERGE, Y1, WIRE, 2.4)
f.arrow(f"M{MERGE} {Y1}H{C3 - 1}", WIRE, 1.3, head=6.5)

HEAD_H1, HEAD_H2 = 106, 156  # output-head card heights, rows 1 and 2
act1 = f.card(C4, Y1 - HEAD_H1 / 2, C4_W, HEAD_H1, fill="#FFFFFF", stroke=HAIR)
VEC_CELL, VEC_GAP = 22, 5
vec_w = 4 * VEC_CELL + 3 * VEC_GAP
vx = C4 + (C4_W - vec_w) / 2
f.text(C4 + C4_W / 2, Y1 - 24, "$a_t$", size=LABEL, anchor="middle", box=act1)
vec = f.vector(vx, Y1 - VEC_CELL / 2, 4, "blue", cell=VEC_CELL, vertical=False, gap=VEC_GAP,
               values=[0.55, 0.35, 0.75, 0.9])
for k, s_ in enumerate(("$\\Delta x$", "$\\Delta y$", "$\\Delta z$", "$g$")):
    f.text(vx + k * (VEC_CELL + VEC_GAP) + VEC_CELL / 2, Y1 + VEC_CELL / 2 + 25, s_, size=MIN, anchor="middle",
           box=act1)
f.connect(wout1, act1)
ik = f.chip(C5 + (C5_W - 56) / 2, Y1 - 18, 56, 36, "IK", "gray", size=LABEL)
f.connect(act1, ik)

# row 2: leg motor neurons -> readout scale s -> W_out -> action chunk -> temporal ensemble
mn2 = neurons(C1, Y2, "68", "leg motor", "blue", C1_W)
f.connect(cns, mn2, sides=("right", "left"), ta=None)
S_W = 32
s_node = f.chip(C1 + C1_W + (C3 - C1 - C1_W - S_W) / 2, Y2 - S_W / 2, S_W, S_W, "$s$", "gray", size=LABEL,
                fill="#FFFFFF", color=INK)
f.asset(SNOW, f.rect(s_node)[0] + S_W / 2 - 9, Y2 - S_W / 2 - 6 - 18, 18, color=None)
f.connect(mn2, s_node, head=6)
wout2 = linear(C3, Y2, "$W_{\\mathrm{out}}$", "right")
f.connect(s_node, wout2, head=6)

chunk = f.card(C4, Y2 - HEAD_H2 / 2, C4_W, HEAD_H2, fill="#FFFFFF", stroke=HAIR)
CELL, GAP = 7, 2
grid_w = 10 * CELL + 9 * GAP
grid_h = 9 * CELL + 8 * GAP
gx, gy = C4 + (C4_W - grid_w) / 2, Y2 - grid_h / 2 + 3
f.text(C4 + C4_W / 2, Y2 - HEAD_H2 / 2 + 27, "$\\hat{a}_{t:t+k}$", size=LABEL, anchor="middle", box=chunk)
fade = [0.95 - 0.07 * c for c in range(10)]
f.token_grid(gx, gy, 9, 10, "blue", cell=CELL, gap=GAP, values=[fade[c] for _ in range(9) for c in range(10)])
f.text(C4 + C4_W / 2, Y2 + HEAD_H2 / 2 - 13, "$k = 10$", size=MIN, anchor="middle", color=MUTED, box=chunk)
f.connect(wout2, chunk)

ens_h = HEAD_H2
ens = f.card(C5, Y2 - ens_h / 2, C5_W, ens_h, fill="#FFFFFF", stroke=HAIR)
f.text(C5 + C5_W / 2, Y2 - ens_h / 2 + 27, "ensemble", size=MIN, weight=500, anchor="middle", box=ens)
# chunks predicted at t-2, t-1 and t, one row each, aligned on absolute time; column t is averaged
EC, EP, ER = 10, 12, 15  # cell, column pitch, row pitch
rows_n, cols_n = 3, 5
sk_w = (cols_n + rows_n - 1) * EP - (EP - EC)
ex0 = C5 + (C5_W - sk_w) / 2
ey0 = Y2 - 30
for r in range(rows_n):
    for c in range(cols_n):
        x, y = ex0 + (c + r) * EP, ey0 + r * ER
        f.add(f'<rect x="{x:.1f}" y="{y:.1f}" width="{EC}" height="{EC}" rx="1.5" fill="{BLU.mid}" '
              f'stroke="{BLU.accent}" stroke-width="0.7"/>')
hx = ex0 + (rows_n - 1) * EP - 2.5
hb = ey0 + (rows_n - 1) * ER + EC + 2.5
f.add(f'<rect x="{hx:.1f}" y="{ey0 - 2.5:.1f}" width="{EC + 5}" height="{hb - ey0 + 2.5:.1f}" rx="2" '
      f'fill="none" stroke="{INK}" stroke-width="1.1"/>')
acx = hx + (EC + 5) / 2
f.arrow(f"M{acx:.1f} {hb:.1f}V{Y2 + 34:.1f}", WIRE, 1.2, head=6)
f.text(acx, Y2 + 58, "$a_t$", size=LABEL, anchor="middle", box=ens)
f.connect(chunk, ens)

# ================================================================ (a) -> (b) -> (c) through the connectome only
f.connect(in1, cns, sides=("right", "left"), tb=None)
f.connect(in2q, cns, sides=("right", "left"), tb=None)
f.connect(in2o, cns, sides=("right", "left"), tb=None)

# ================================================================ closed loop through MuJoCo
LOOP = dict(color=FAINT, sw=1.3, dashed=True, head=7, knockout=True, label_size=MIN)
f.route(ik, img1, (XR, LANE_T, LANE_L), sides=("right", "left"), label="MuJoCo", label_seg=2,
        label_at=XB + WB / 2, **LOOP)
f.route(ens, img2, (XR, LANE_B, LANE_L), sides=("right", "left"), label="MuJoCo", label_seg=2,
        label_at=XB + WB / 2, **LOOP)

# ================================================================ (d) causal controls
pd = f.panel(XA, SY0, XC + WC - XA, SY1 - SY0, "gray", "(d) Controls")
GROUPS = (("baselines", "gray", ("shuffled CNS", "GRU", "MLP")),
          ("lesions", "gray", ("edges off", "direct only", "deafferented", "state reset")))
PADX, INNER, GROUP_GAP, CH = 26, 10, 40, SY1 - SY0 - 16
runs = []
for label, role, names in GROUPS:
    items = [("label", label, text_w(label, MIN) + 2)] + [("chip", n, text_w(n, LABEL) + PADX) for n in names]
    runs.append((role, items))
x0 = XA + 12 + text_w("(d) Controls", TITLE, bold=True) + 40
x1 = XC + WC - 14
n_chips = sum(kind == "chip" for _, items in runs for kind, _, _ in items)
n_items = sum(len(items) for _, items in runs)
free = x1 - x0 - sum(w for _, items in runs for _, _, w in items) - INNER * (n_items - len(runs)) - GROUP_GAP
extra = max(0.0, free / n_chips)
x = x0
for role, items in runs:
    for kind, s_, w in items:
        if kind == "label":
            f.text(x, SY0 + 30, s_, size=MIN, color=MUTED, family="serif", italic=True, box=pd)
        else:
            w += extra
            f.chip(round(x), SY0 + 8, round(w), CH, s_, role, size=LABEL, fill="#FFFFFF")
        x += w + INNER
    x += GROUP_GAP - INNER
print(f"controls: extra chip padding {extra:.1f}px")

f.save(str(OUT))
print(OUT)
