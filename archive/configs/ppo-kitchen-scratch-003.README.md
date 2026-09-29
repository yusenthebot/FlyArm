# ppo-kitchen-scratch-003

Reward-only kitchen run after runs/ppo-kitchen-scratch-001 and -002 both collapsed on their
first policy update (research log E39).
Five changes against configs/ppo-kitchen-scratch.json.
Ablate in this order, strongest evidence last.

| # | change | from | to | evidence |
|---|--------|------|----|----------|
| 5 | `readout_calibration` | `scale` | `unit_norm` | the root cause, and the only change with a controlled experiment behind it: everything else identical, the first trained iteration's mean ratio deviation is 0.930 with `scale` and 0.255 with `unit_norm`, and `unit_norm` then holds a steady 0.17 to 0.18 instead of spiking and freezing |
| 2 | `advantage_clip` | 0 (off) | 10 | the general guard against a heavy advantage tail, commit b86e615; it did not fix -002 on its own, because that blow-up was in the policy head, not the advantages |
| 1 | `critic_warmup` | 5 | 50 | the critic should predict the 200 bonus before the policy chases it; -002 shows it is not sufficient on its own |
| 3 | `train_variant.initial_joint_offset` | 0.1 | 0.3 | every kitchen episode starts from the same state (E29), so the 128 environments were near copies; the tracker still scores 100 at 0.3 rad (E30) |
| 4 | `approach_slope` | 3.0 | 1.0 | raises the per-step floor from 0.0124 to 0.1086, cutting the bonus-to-floor ratio from 16,115:1 to 1,842:1, at the cost of dropping the tracker-to-random discrimination from 13.6x to 2.4x |

Change 4 is the weakest and should be the first reverted once the run works.

## Why `unit_norm` is the root cause

The policy head is `tanh(decoder(readout))` over 2,022 readout outputs.
With `scale` those outputs have mean magnitude 0.217, so the decoder pre-activation is only
0.121 on average and 0.410 at most, while one PPO iteration (4 epochs x 4 minibatches at
learning rate 3e-4) shifts every decoder weight by about `16 * 3e-4` and therefore shifts the
pre-activation by about `16 * 3e-4 * 0.217 * 2022 = 2.11`.
The first trained iteration drives the tanh straight into saturation, the mean action snaps to
a corner, and the policy never comes back.
`unit_norm` divides the readout by `sqrt(2022)`, which makes the same shift 0.08 and keeps the
head in its linear region.
This is research log E32's argument ("at n = 1,382 one update saturates the tanh, and the
1 / sqrt(n) factor prevents it") at n = 2,022; the earlier configs used `scale` only because
they mirrored the pick-and-place `scratch_policy`.

A warm-started run needs no such setting: it loads the imitation checkpoint's own frozen
`readout_scale`, and runs/flyleg-kitchen-body-chunk1-001 is trained with `unit_norm`.
