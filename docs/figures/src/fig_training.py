"""Figure: training the connectome controller on the manipulation benchmark.

(a) Skill-level DAgger: every round resets batched environments to recorded subgoal starts or to
true starts, rolls out the learner (mixed with the teacher at rate beta in the first rounds), labels
every visited state with the privileged teacher, aggregates the labels and updates the encoder and
decoder by truncated backpropagation through the frozen connectome; three curriculum stages.
(b) PPO with a demonstration term: 128 batched environments, rollouts through the frozen connectome,
a privileged critic and GAE, clipped updates for the decoder and for the encoder (through one
control step), DAPG terms for both from demonstration states replayed every 25 iterations, and
checkpoint selection on validation episodes only.

Facts: src/flyarm/manipulation/skill_dagger.py, src/flyarm/rl/ppo.py, src/flyarm/rl/ppo_manipulation.py,
configs/skill-dagger-connectome.json, configs/ppo-manipulation-encoder-clipped.json, docs/RESEARCH_LOG.md
(E55 to E61). Run from any directory:

    python fig_training.py
    python qa_svg_figure.py ../training-figure.svg --png --strict-tidy
"""

from pathlib import Path

from figkit import FAINT, HAIR, MUTED, PAL, Fig

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "training-figure.svg"
ASSETS = HERE / "assets"
FIRE, SNOW = ASSETS / "fire.svg", ASSETS / "snowflake.svg"

W, H = 1400, 392
f = Fig(W, H)
LABEL, MODULE, MIN = f.fs("label"), f.fs("module"), f.fs("min")
MARK = 20

PY0, PY1 = 28, 384
WA, WB = 664, 684
XA, XB = f.place(20, 1380, (WA, WB), gap=12)
Y = 176  # row line of both flows
NH = 76  # node height
LOOP = dict(color=FAINT, sw=1.3, dashed=True, head=7, knockout=True, label_size=MIN)


def node(x, w, title, sub, role=None, mark=None):
    c = f.card(x, Y - NH / 2, w, NH, role, fill=None if role else "#FFFFFF", stroke=None if role else HAIR)
    f.text(x + w / 2, Y - 8, title, size=LABEL, weight=500, anchor="middle", box=c)
    f.text(x + w / 2, Y + 20, sub, size=MIN, color=MUTED, anchor="middle", box=c)
    if mark is not None:
        f.asset(mark, x + w - MARK - 6, Y - NH / 2 + 6, MARK, color=None)
    return c


# ================================================================ (a) skill-level DAgger
f.panel(XA, PY0, WA, PY1 - PY0, "amber", "(a) Skill-level DAgger")
AW = (112, 116, 108, 96, 116)
ax = f.place(XA + 12, XA + WA - 12, AW, gap=20)
bank = f.cylinder(ax[0], Y - NH / 2, AW[0], NH, "gray", "subgoal bank", size=MIN)
roll = node(ax[1], AW[1], "Rollout", "learner, $\\beta$", "amber")
lab = node(ax[2], AW[2], "Labels", "teacher $\\pi^{\\star}$")
data = f.cylinder(ax[3], Y - NH / 2, AW[3], NH, "gray", "$\\mathcal{D}$", size=LABEL)
upd = node(ax[4], AW[4], "Update", "$E_\\phi$, $W_{\\mathrm{out}}$", "red")
for a, b in zip((bank, roll, lab, data), (roll, lab, data, upd)):
    f.connect(a, b)
f.route(upd, roll, (Y - NH / 2 - 34,), sides=("top", "top"), label="next round", **LOOP)

# curriculum: three stages of rounds
CY = 280
f.text(XA + 24, CY - 14, "curriculum", size=MIN, color=MUTED, family="serif", italic=True)
STAGES = (("1 subgoal", "$\\times 4$"), ("2\u20133 subgoals", "$\\times 3$"), ("full task", "$\\times 3$"))
SW = (160, 196, 160)
sx = f.place(XA + 12, XA + WA - 12, SW, gap=40)
chips = []
for (name, n), x, w in zip(STAGES, sx, SW):
    c = f.card(x, CY, w, 64, fill="#FFFFFF", stroke=PAL["amber"].accent)
    f.text(x + w / 2, CY + 26, name, size=LABEL, anchor="middle", box=c)
    f.text(x + w / 2, CY + 50, n, size=MIN, color=MUTED, anchor="middle", box=c)
    chips.append(c)
for a, b in zip(chips, chips[1:]):
    f.connect(a, b)

# ================================================================ (b) PPO with DAPG
f.panel(XB, PY0, WB, PY1 - PY0, "blue", "(b) PPO with DAPG")
BW = (116, 132, 124, 212)
bx = f.place(XB + 12, XB + WB - 12, BW, gap=24)
env = node(bx[0], BW[0], "MuJoCo", "128 envs")
cns = node(bx[1], BW[1], "Rollout", "frozen CNS", "gray", SNOW)
adv = node(bx[2], BW[2], "GAE", "$\\hat{A}_t$, $\\gamma = 0.995$")
UH = 150
upd2 = f.card(bx[3], Y - NH / 2, BW[3], UH, "red")
f.asset(FIRE, bx[3] + BW[3] - MARK - 6, Y - NH / 2 + 6, MARK, color=None)
f.text(bx[3] + 14, Y - NH / 2 + 26, "Clipped update", size=LABEL, weight=500, box=upd2)
for k, s_ in enumerate(("decoder $W_{\\mathrm{out}}$", "encoder $E_\\phi$, 1 step")):
    f.chip(bx[3] + 12, Y - NH / 2 + 42 + 50 * k, BW[3] - 24, 40, s_, "red", size=MIN, fill="#FFFFFF")
f.connect(env, cns)
f.connect(cns, adv)
f.connect(adv, upd2, sides=("right", "left"), tb=None)
f.route(upd2, env, (Y - NH / 2 - 34,), sides=("top", "top"), ta=0.5, label="next iteration", **LOOP)

# supports under the flow: the privileged critic and the demonstrations
DY = 280
crit = f.card(bx[2], DY, BW[2], 64, fill="#FFFFFF", stroke=HAIR)
f.text(bx[2] + BW[2] / 2, DY + 26, "Critic $V$", size=LABEL, anchor="middle", box=crit)
f.text(bx[2] + BW[2] / 2, DY + 50, "privileged", size=MIN, color=MUTED, anchor="middle", box=crit)
f.connect(crit, adv)
demo = f.cylinder(bx[3], DY + 20, BW[3], 72, "gray", "demo states", size=MIN)
f.connect(demo, upd2)
sel = f.card(bx[0], DY, BW[0] + 24 + BW[1], 64, fill="#FFFFFF", stroke=HAIR, dashed=True)
f.text(bx[0] + (BW[0] + 24 + BW[1]) / 2, DY + 26, "Selection", size=LABEL, anchor="middle", box=sel)
f.text(bx[0] + (BW[0] + 24 + BW[1]) / 2, DY + 50, "validation only", size=MIN, color=MUTED,
       anchor="middle", box=sel)

f.save(str(OUT))
print(OUT)
