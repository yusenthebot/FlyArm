# ppo-kitchen-scratch-005

runs/ppo-kitchen-scratch-003 plus a potential-based reference term.
Everything else is identical, so -003 is the control and the only difference is the term.

Still reward only: the term reads the joint **positions** of one benchmark demonstration and
never its actions, so nothing is cloned anywhere in the pipeline.

| field | value |
|-------|-------|
| `tracking_form` | `potential` |
| `tracking_weight` | 2.0 |
| `reference_episode` | 0 (212 steps of the complete split, fixed by index) |

The term is `tracking_weight * (phi(s') - phi(s))` with `phi(s) = -||q - q_ref||` over the 9
robot joints, that is the weight times the distance closed this step, in radians.

## Why this form

`phi` is undiscounted on purpose. The literal Ng, Harada and Russell form,
`gamma * phi(s') - phi(s)`, carries a `(1 - gamma) * ||q - q_ref||` residual that the agent
collects every step, and measured at weight 1.0 from 0.3 rad starts it pays a random policy
**+3.758** per episode against the demonstration tracker's **+1.369** — more for being far from
the reference than for tracking it. The undiscounted form pays the tracker **+0.091** and the
random policy **-3.255**, which is the right way round. It is kept as
`tracking_form: "potential_discounted"` with that measurement recorded, and the Gaussian kernel
of E41 as `"gaussian"`; neither should be reached for.

Properties of the form actually used:

- **Telescopes exactly.** The undiscounted sum over an episode is `phi(s_T) - phi(s_0)`,
  whatever path is taken, which a test asserts on two different paths.
- **Standing still pays exactly 0**, so it cannot be farmed by hovering.
- **Leaves the E34 floor alone.** It does not enter the per-step budget, so the per-step maximum
  stays 1.0 and the completion bonus of 200 keeps its full 2.0x headroom. Only the Gaussian form
  enters the budget, and the config enforces that distinction.
- **Alive at any distance.** Shaving 0.1 rad pays the same at 5 rad as at 0.5 rad, which is
  exactly what the Gaussian kernel could not do.

## Why weight 2.0

Measured, potential form, 0.3 rad starts:

| | distance | per-step \|term\| | episode total | total shaping per step | tasks |
|---|---|---|---|---|---|
| tracker | 0.44 → 0.35 | 0.075 | **+0.18** | 0.265 | 4.00 |
| random | 0.45 → 3.70 | 0.096 | **-6.51** | 0.085 | 0.00 |

It puts the term's per-step magnitude on the same scale as the task shaping a learning policy
actually earns (0.109 per step), and the most it can add over a whole episode is
`weight x starting distance` = 0.90, which is **0.45% of one completion bonus**, so it guides
without ever competing with task completion. Total shaping discrimination rises from 2.4x to
**3.1x**.

Note how it earns that: the term barely rewards the expert (+0.0008 per step) and substantially
penalises drift (-0.023 per step). It is a drift penalty, not a tracking bonus, which is what
potential shaping reduces to when the expert already starts near the reference.

## Guards, unchanged from -004

E29 shows the clean kitchen is solvable by blind replay, so any reference term could in
principle be maximised open loop. Training starts are perturbed by 0.3 rad, where E30 measures
replay at 16 to 19 against 100 for closed-loop control, and the reported checkpoint is selected
on perturbed validation episodes only. The nominal score is reported but never selected on.
