# ppo-kitchen-scratch-004

runs/ppo-kitchen-scratch-003 plus a dense reference-tracking reward, in the spirit of
DeepMimic and AMP.
Everything else is identical, so -003 is the no-reference control for this run.

Still reward only: the reward reads the joint **positions** of one benchmark demonstration and
never its actions, so no behaviour is cloned anywhere in the pipeline.

| field | value | why |
|-------|-------|-----|
| `tracking_weight` | 0.5 | the largest weight with real headroom under the E34 invariant: the per-step maximum becomes 1.5, whose discounted value at gamma 0.99 is 150 against a completion bonus of 200. A weight of 1.0 puts the floor at 200 and 1.5 at 250, which the constructor and the config both refuse. |
| `tracking_sigma` | 0.6 rad | chosen by measurement, see below |
| `reference_episode` | 0 | 212 steps of the complete split; fixed by index so the run is reproducible |

## Why sigma 0.6

Measured joint error against the reference, 280-step episodes:

| | `\|\|q - q_ref\|\|` mean | p90 |
|---|---|---|
| demonstration tracker, nominal start | 0.003 | 0.004 |
| demonstration tracker, 0.3 rad start | 0.584 | 1.038 |
| random policy, either start | 2.5 | 3.8 |

Sigma trades discrimination against a usable gradient from the 0.3 rad starts this run trains
from. A typical 0.3 rad draw puts the arm about 0.46 rad from the reference at step 0, so
sigma 0.6 pays `exp(-0.46^2 / 0.6^2) = 0.56` immediately, which is reward the policy can
preserve rather than a flat zero it has to find. Smaller sigma discriminates harder but starts
near zero (0.095 at sigma 0.3), larger sigma is too forgiving (tracker to random falls from
18.8x at sigma 0.6 to 4.8x at sigma 1.5).

## What the term is worth

| | tracking term | x 0.5 | total shaping per step |
|---|---|---|---|
| tracker, nominal | 1.000 | 0.500 | 0.783 |
| tracker, 0.3 rad | 0.469 | 0.235 | 0.499 |
| random, nominal | 0.031 | 0.016 | 0.119 |
| random, 0.3 rad | 0.025 | 0.012 | 0.121 |

On the training distribution the reference term separates the tracker from a random policy by
18.8x on its own, and it lifts the total shaping discrimination from 2.4x to **4.1x**.

## The risk it carries, and the two guards already in place

E29 shows the clean kitchen is solvable by blind replay, so a time-indexed reference reward
could in principle be maximised open loop. Two things in this config prevent that from being
the easy answer: training starts are perturbed by 0.3 rad, where E30 measures replay at 16 to
19 against 100 for closed-loop control, and the reported checkpoint is selected on perturbed
validation episodes only. The nominal score is reported but never selected on.
