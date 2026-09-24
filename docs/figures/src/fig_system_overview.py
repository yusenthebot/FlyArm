"""Figure: FlyArm system overview - the control loop through the frozen fly CNS, the four MuJoCo tasks, the
three training stages and the causal controls.

(a) One control step. The normalized observation o_t is mapped by a trained linear encoder W_in to input currents
    I_t injected into the 1,846 ascending sensory neurons of the frozen MaleCNS v1.0 connectome (166,700 neurons,
    10.5 M synapses, weights = measured synapse counts). Three neural steps h <- 0.5 h + 0.5 tanh(I + g W h), g = 0.8,
    run per control step; the readout is the mean rate of 2,022 neurons (1,314 descending, 708 VNC motor) over
    those steps, calibrated by a frozen standardization divided by sqrt(2022), and a trained linear decoder W_out
    with tanh gives a_t. The state h persists across control steps. MLX on the Apple GPU, about 4 ms per control
    step for one environment and 20 ms for 128. This is the B1a whole-body interface used by every main result.
(b) Tasks, MuJoCo with the Franka Panda, all batched by mjbatch (N MuJoCo copies on a C++ thread pool, step-exact
    against a single environment).
(c) Training: imitation, then PPO warm-started from it, then evaluation.
(d) Controls: a degree-preserving shuffle of the connectome and parameter-matched GRU and MLP; lesions of a
    trained policy.

The connectome is drawn as a render and a single equation, never as a multi-hop propagation diagram: on the
manipulation benchmark it acts as one fixed synaptic projection (deep-path signal about 1e-3, shuffle equivalent),
which the caption states.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from figkit import HAIR, INK, MUTED, PAL, WIRE, Fig, text_w

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "system_overview-figure.svg"
A = HERE / "assets"

W, H = 1400, 732
f = Fig(W, H)
TITLE, MODULE, LABEL, SMALL = f.fs("title"), f.fs("module"), f.fs("label"), f.fs("min")
AMB, BLU, RED, GRN, GRY = (PAL[k] for k in ("amber", "blue", "red", "green", "gray"))
LOOP = "#7A7A7A"


def link(a, b, sides=("right", "left"), **kw):
    kw.setdefault("color", WIRE)
    kw.setdefault("sw", 1.3)
    kw.setdefault("head", 7)
    return f.connect(a, b, sides=sides, **kw)


def lines(x, y, items, box, color=INK, pitch=22, family="sans", italic=False, size=None):
    for i, s in enumerate(items):
        f.text(x, y + pitch * i, s, size=size or SMALL, color=color, box=box, family=family, italic=italic)


def mark(kind, x, y, px=18):
    f.asset(A / ("fire.svg" if kind == "trained" else "snowflake.svg"), x, y, px, color=None)


def run(x0, widths, gaps):
    """x of each item in a left-to-right run with explicit gaps."""
    xs, x = [], x0
    for w, g in zip(widths, (*gaps, 0), strict=True):
        xs.append(x)
        x += w + g
    return xs


# ================================================================ panels
PA = (8, 8, 1384, 300)
PB = (8, 316, 592, 336)
PC = (608, 316, 784, 336)
PD = (8, 660, 1384, 64)
f.panel(*PA, "gray", "(a) Control loop", sub="one control step", title_size=TITLE)
f.panel(*PB, "gray", "(b) Tasks", sub="observation $\\to$ action, horizon", title_size=TITLE)
f.panel(*PC, "gray", "(c) Training", sub="three stages", title_size=TITLE)
f.panel(*PD, "gray", None)

# ================================================================ (a) control loop
FA = 170  # the flow line
CNS_Y, CNS_H = 70, 200
LANE = 290
WIDTHS = (14, 58, 60, 124, 422, 166, 130, 60, 56, 14)
GAPS = (26, 26, 40, 26, 26, 26, 26, 26, 26)  # the wider gap carries $I_t$
X = run(24 + (1352 - sum(WIDTHS) - sum(GAPS)) / 2, WIDTHS, GAPS)
OX, NX, WIX, ASX, CX, RX, KX, WOX, TX, AX = X

obs = f.vector(OX, FA - (8 * 14 + 7) / 2, 8, "gray", cell=14)
f.text(OX + 7, FA - 70, "$o_t$", size=LABEL, anchor="middle", color=INK)
norm = f.chip(NX, FA - 18, 58, 36, "norm", size=SMALL, fill="#FFFFFF", stroke=HAIR, color=INK)
w_in = f.trapezoid(WIX, FA - 35, 60, 70, "red", s="$W_{\\mathrm{in}}$", size=LABEL, direction="left")
mark("trained", WIX + 21, FA - 62)
asc = f.card(ASX, FA - 38, 124, 76, "amber")
f.text(ASX + 62, FA - 6, "1,846", size=MODULE, weight=600, anchor="middle", color=AMB.deep, box=asc)
f.text(ASX + 62, FA + 20, "ascending", size=SMALL, anchor="middle", color=AMB.deep, box=asc)

cns = f.card(CX, CNS_Y, 422, CNS_H, fill="#FFFFFF", key=True)
f.image(A / "malecns_brain.png", CX + 12, FA - 84, 200, 168, fit="contain", frame=None)
TX0 = CX + 228
f.text(TX0, CNS_Y + 32, "MaleCNS v1.0", size=MODULE, weight=600, color=INK, box=cns)
mark("frozen", TX0 + text_w("MaleCNS v1.0", MODULE, bold=True) + 10, CNS_Y + 15)
lines(TX0, CNS_Y + 60, ("166,700 neurons", "10.5M synapses"), cns, color=MUTED)
f.text(TX0, CNS_Y + 112, "$h \\gets 0.5\\,h$", size=LABEL, color=INK, box=cns, family="serif")
f.text(TX0, CNS_Y + 142, "$+\\,0.5\\,\\tanh(I + 0.8\\,Wh)$", size=LABEL, color=INK, box=cns, family="serif")
f.text(TX0, CNS_Y + 176, "$\\times 3$ per step, mean rate", size=SMALL, color=MUTED, box=cns)
f.arc(cns, cns, sides=("top", "top"), ta=0.72, tb=0.9, bulge=-26, color=LOOP, dashed=False, head=6.5,
      label="$h$ persists", label_size=SMALL)

readout = f.card(RX, FA - 65, 166, 130, "blue")
f.text(RX + 12, FA - 39, "2,022 read out", size=SMALL, weight=600, color=BLU.deep, box=readout)
for k, s in enumerate(("1,314 descending", "708 VNC motor")):
    f.chip(RX + 12, FA - 22 + k * 40, 142, 32, s, size=SMALL, fill="#FFFFFF", stroke=BLU.mid, color=INK)
calib = f.card(KX, FA - 36, 130, 72, fill="#FFFFFF", stroke=HAIR)
f.text(KX + 12, FA - 8, "standardize", size=SMALL, color=INK, box=calib)
f.text(KX + 12, FA + 20, "÷ $\\sqrt{2022}$", size=SMALL, color=INK, box=calib)
mark("frozen", KX + 130 - 26, FA - 30, 16)
w_out = f.trapezoid(WOX, FA - 35, 60, 70, "red", s="$W_{\\mathrm{out}}$", size=LABEL, direction="right")
mark("trained", WOX + 21, FA - 62)
tanh = f.chip(TX, FA - 18, 56, 36, "tanh", size=SMALL, fill="#FFFFFF", stroke=HAIR, color=INK, family="mono")
act = f.vector(AX, FA - (5 * 14 + 4) / 2, 5, "red", cell=14)
f.text(AX + 7, FA - 48, "$a_t$", size=LABEL, anchor="middle", color=INK)

for a, b in pairwise((obs, norm, w_in, asc, cns, readout, calib, w_out, tanh, act)):
    link(a, b, head=6.5)
f.text((WIX + 60 + ASX) / 2, FA - 8, "$I_t$", size=SMALL, anchor="middle", color=INK)
f.route(act, obs, (LANE,), sides=("bottom", "bottom"), color=LOOP, sw=1.4, dashed=True, head=7, label="MuJoCo",
        label_seg=1, label_at=CX + 215, label_color=MUTED, label_size=SMALL, knockout=True)

# ================================================================ (b) tasks
TASKS = (
    ("render-pick-place.png", "Pick-place", "37 $\\to$ 4", "400 steps", "a block"),
    ("render-kitchen.png", "FrankaKitchen", "30 $\\to$ 9", "280 $\\times$ 80 ms", "4 subtasks"),
    ("render-grasp-objects.png", "Grasp", "54 $\\to$ 5", "~200 steps", "41 objects"),
    ("render-manipulation.png", "Manipulation", "220 $\\to$ 5", "1,100-2,600", "11 templates"),
)
CARD_W, CARD_H = 274, 118
B_X = f.place(PB[0] + 16, PB[0] + PB[2] - 16, (CARD_W, CARD_W))
B_Y = (PB[1] + 58, PB[1] + 58 + CARD_H + 12)
for k, (img, name, dims, horizon, extra) in enumerate(TASKS):
    x, y = B_X[k % 2], B_Y[k // 2]
    card = f.card(x, y, CARD_W, CARD_H, fill="#FFFFFF", stroke=HAIR)
    f.image(A / img, x + 10, y + (CARD_H - 84) / 2, 112, 84)
    f.text(x + 134, y + 26, name, size=SMALL, weight=600, color=INK, box=card)
    lines(x + 134, y + 50, (dims, horizon), card, pitch=24)
    f.text(x + 134, y + 100, extra, size=SMALL, color=MUTED, box=card, family="serif", italic=True)
f.chip(PB[0] + PB[2] - 16 - 92, PB[1] + 20, 92, 28, "mjbatch", size=SMALL, fill=GRY.tint, stroke=GRY.accent,
       color=INK, family="mono")

# ================================================================ (c) training
C_Y, C_H = PC[1] + 58, 262
S_W = (176, 350, 156)
S_X = f.place(PC[0] + 16, PC[0] + PC[2] - 16, S_W)
stages = []
for x, w, (tag, name) in zip(S_X, S_W, (("A", "Imitation"), ("B", "PPO"), ("C", "Evaluation")), strict=True):
    card = f.card(x, C_Y, w, C_H, fill="#FFFFFF", stroke=HAIR, key=tag == "B")
    f.text(x + 12, C_Y + 28, f"{tag}  {name}", size=LABEL, weight=600, color=INK, box=card)
    stages.append(card)
link(stages[0], stages[1], ta=28 / C_H, tb=28 / C_H)
link(stages[1], stages[2], ta=28 / C_H, tb=28 / C_H)

ax = S_X[0] + 12
RW_TOP = C_Y + 70  # first reward chip; A and B's left column share its 30 px row lines
f.text(ax, C_Y + 56, "scripted teacher", size=SMALL, color=MUTED, family="serif", italic=True, box=stages[0])
f.text(ax, RW_TOP + 19, "demo tracker", size=SMALL, color=MUTED, family="serif", italic=True, box=stages[0])
lines(ax, RW_TOP + 60 + 19, ("BC + DAgger", "truncated BPTT", "skill-balanced", "closed-loop val"), stages[0],
      pitch=30)

bx = S_X[1] + 12
f.text(bx, C_Y + 56, "warm start", size=SMALL, color=MUTED, family="serif", italic=True, box=stages[1])
lines(bx, RW_TOP + 19, ("actor = (a)", "trains $W_{\\mathrm{out}}$"), stages[1], pitch=30)
f.text(bx, RW_TOP + 60 + 19, "critic", size=SMALL, color=INK, box=stages[1])
f.mlp(bx + 56, RW_TOP + 60, 88, 26, (3, 4, 1), "gray", r=3)
lines(bx, RW_TOP + 90 + 19, ("privileged", "GAE, clip", "+ DAPG"), stages[1], color=MUTED, pitch=30)
REWARD = ("ordered completion", "progress potential", "action quality", "strict completion", "state resets",
          "curriculum")
RW_X, RW_W = S_X[1] + 350 - 12 - 176, 176
f.text(RW_X, C_Y + 56, "reward", size=SMALL, weight=600, color=GRN.deep, box=stages[1])
for k, s in enumerate(REWARD):
    f.chip(RW_X, RW_TOP + k * 30, RW_W, 26, s, size=SMALL, fill=GRN.tint, stroke=GRN.mid, color=GRN.deep)

cx = S_X[2] + 12
lines(cx, C_Y + 56, ("official envs", "strict metrics", "val checkpoint"), stages[2], pitch=24)
f.text(cx, C_Y + 136, "held out", size=SMALL, color=MUTED, family="serif", italic=True, box=stages[2])
lines(cx, C_Y + 162, ("perturbed starts", "unseen objects", "unseen furniture", "unseen combos"), stages[2], pitch=24)

# ================================================================ (d) controls and lesions
D_MID = PD[1] + PD[3] / 2
f.text(PD[0] + 16, D_MID + 8, "(d) Controls", size=TITLE, weight=700, color=INK)
GROUPS = (("baselines", ("shuffled CNS", "GRU", "MLP")),
          ("lesions", ("edges off", "direct only", "state reset", "deafferented")))
x = PD[0] + 16 + text_w("(d) Controls", TITLE, bold=True) + 36
for label, items in GROUPS:
    f.text(x, D_MID + 6, label, size=SMALL, color=MUTED, family="serif", italic=True)
    x += text_w(label, SMALL) + 16
    for s in items:
        w = text_w(s, SMALL) + 28
        f.chip(x, D_MID - 16, w, 32, s, size=SMALL, fill="#FFFFFF", stroke=HAIR, color=INK)
        x += w + 10
    x += 26
LEG_X = PD[0] + PD[2] - 16  # trained and frozen marks, right-aligned in the strip
f.text(LEG_X, D_MID + 6, "frozen", size=SMALL, color=MUTED, anchor="end")
mark("frozen", LEG_X - text_w("frozen", SMALL) - 24, D_MID - 9)
f.text(LEG_X - text_w("frozen", SMALL) - 40, D_MID + 6, "trained", size=SMALL, color=MUTED, anchor="end")
mark("trained", LEG_X - text_w("frozen", SMALL) - text_w("trained", SMALL) - 64, D_MID - 9)

f.save(str(OUT))
print(OUT)
