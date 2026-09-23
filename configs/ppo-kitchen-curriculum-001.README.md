# ppo-kitchen-curriculum-001

The attack on one task. `runs/ppo-kitchen-scratch-003` stays the control.

Two changes against `-003`, both measured first (research log E44 and E45).

| field | control | here | why |
|-------|---------|------|-----|
| `train_variant.curriculum_prefix` | 0 | 3 | reward alone reaches the kettle and stops; the curriculum puts the policy in states where a second, third and fourth element is the next thing to do |
| `train_variant.curriculum_true_start_share` | n/a | 0.25 | a quarter of environments always start at the true beginning, so the real start keeps being practised |
| `task_shaping_form` | `level` | `potential` | the level form pays for standing still and a random policy out-earns the expert at two of four prefixes |
| `task_shaping_weight` | 1.0 | 10.0 | scale, see below |

Evaluation is unchanged and never uses the curriculum: the unmodified benchmark from the
true start, on `-003`'s seeds, with the reported checkpoint selected on perturbed validation
episodes only.

## Why the potential form, in one line

It is the only one of the three measured forms whose expert-to-idle separation holds at every
curriculum prefix: the expert earns a positive term at all four prefixes and a random policy a
negative one, where bonus-only has no separation at all (0 against 0) and the level form has the
expert losing to random at two of four.

## The table that decided it

Per-step shaping, expert against random **within the same prefix**, 32 environments each, 0.3 rad
starts, `curriculum_prefix` 3.

| prefix | bonus only | potential | level (control's form) |
|---|---|---|---|
| | expert / random | expert / random | expert / random |
| 0 | 0.0000 / 0.0000 | **+0.0116 / -0.0002** | 0.2084 / 0.1097 (1.90x) |
| 1 | 0.0000 / 0.0000 | **+0.0084 / -0.0001** | 0.1389 / **0.1643 (0.85x)** |
| 2 | 0.0000 / 0.0000 | **+0.0059 / -0.0004** | 0.1056 / 0.0712 (1.48x) |
| 3 | 0.0000 / 0.0000 | **+0.0031 / -0.0003** | 0.0483 / **0.0607 (0.80x)** |

The expert earns the maximum earnable from every prefix in all three cases (4.00, 3.00, 2.00,
1.00 tasks), so the curriculum's states are fully solvable and the differences above are the
reward's, not the states'.

## Why weight 10

At weight 1 the potential term's whole-episode total for the expert is 2.01, which is 1.0% of a
single completion bonus and too small to guide. At weight 10 it is 20.07, that is **10.0% of one
bonus** (largest observed 32.17, or 16%), while a random policy collects -0.99. The total is
bounded above by `segments x MAX_STEP_REWARD x weight = 4 x 1.0 x 10 = 40`, or 20% of one bonus,
so it guides without ever competing with completing a task. The E34 floor is untouched because
the potential form telescopes and does not enter the per-step maximum.

## What the curriculum assumes

Pre-completed elements are placed at their goal joint positions with zero velocity. That state
is reachable in principle, since each goal is a configuration the arm can put the element in, but
it is not a state this policy reached and the arm is not posed as it would be had the policy got
there itself. This is why evaluation never uses it.
