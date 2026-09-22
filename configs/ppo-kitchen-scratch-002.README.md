# ppo-kitchen-scratch-002

Rerun of the reward-only kitchen run after runs/ppo-kitchen-scratch-001 collapsed on its first
policy update (research log E39).
Four changes against configs/ppo-kitchen-scratch.json, all in at once; if the run works, ablate
them in this order, weakest first.

| # | change | from | to | why |
|---|--------|------|----|-----|
| 1 | `critic_warmup` | 5 | 50 | the critic must predict the 200 bonus before the policy chases it |
| 2 | `advantage_clip` | 0 (off) | 10 | bounds the first update; the general guard, commit b86e615 |
| 3 | `train_variant.initial_joint_offset` | 0.1 | 0.3 | every kitchen episode starts from the same state (E29), so the 128 environments were near copies; the tracker still scores 100 at 0.3 rad (E30) |
| 4 | `approach_slope` | 3.0 | 1.0 | raises the per-step floor from 0.0124 to 0.1086, cutting the bonus-to-floor ratio from 16,115:1 to 1,842:1 |

Change 4 also costs discrimination: the tracker-to-random shaping ratio falls from 13.6x to 2.4x
(E39 has the full slope table), so it is the weakest of the four and the first to ablate.
Everything else matches configs/ppo-kitchen-scratch.json.
