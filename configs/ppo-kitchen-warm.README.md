# ppo-kitchen-warm

The third arm of the kitchen comparison: imitation, then reward.

| run | starting policy | reference term |
|-----|-----------------|----------------|
| `ppo-kitchen-scratch-003` | random encoder and decoder, reward only | none |
| `ppo-kitchen-scratch-005` | random encoder and decoder, reward only | potential, weight 2.0 |
| `ppo-kitchen-warm` | `runs/flyleg-kitchen-body-chunk1-001/flyleg-0`, imitation | none |

Every reward and optimiser setting is identical to `-003`, so the only difference is where the
policy starts. `iterations` is 1000 rather than 2000 by instruction, which is a budget and does
not affect a comparison at matched iterations.

## Settings realigned to the control

This config was first written before three findings landed, and carried the pre-E39 values.
They are now `-003`'s:

| field | was | now | why |
|-------|-----|-----|-----|
| `approach_slope` | 3.0 | 1.0 | E39; `-003` uses 1.0 |
| `advantage_clip` | 0 (absent) | 10.0 | E39; the general guard |
| `critic_warmup` | 5 | 50 | see below |
| `train_variant.initial_joint_offset` | 0.1 | 0.3 | E39; `-003` trains at 0.3 |

`readout_calibration` is set to `unit_norm` only so that a diff against the control shows
nothing spurious: the field is read by `scratch_kitchen_policy` alone and is unused on this
path. A warm start loads the checkpoint's own frozen `readout_scale`, and this base run was
trained with `unit_norm`, so the saturation of E41 cannot happen here.

## The critic_warmup decision, which reverses an earlier argument of mine

I earlier argued for keeping `critic_warmup` at 5 here, on the grounds that a competent
checkpoint sees completions routinely so the first update is safe. That argument rested on
believing the `-001` and `-002` collapses came from an uncalibrated critic. E41 showed the cause
was tanh saturation in the policy head instead, so the reason for choosing 5 no longer holds.

At 50 the setting is now better on both remaining grounds: it matches the control, and a
calibrated critic before the first policy update is *more* protective of an expensive imitation
checkpoint, not less. The cost is 50 of 1000 iterations.

If the intent was specifically to test `critic_warmup` 5 on a warm start, that is a separate run
and this one should stay matched to the control.

## Reference points for reading the result

The imitation checkpoint's own score, from `runs/flyleg-kitchen-body-chunk1-001/results.json`:
nominal **31.25** (1.25 of 4 tasks), 0.1 rad starts 20.0, **0.2 rad starts 15.0**, and the
edges-off lesion 25.0 (the constant-action artefact of E11 and E32, which opens the microwave in
every clean episode).

`-003`'s best reward-only point, iteration 250: nominal 25.0, perturbed 25.0, validation 25.0.

Note the split already visible between them: imitation is ahead on the clean benchmark
(31.25 against 25.0) and behind under perturbed starts (15.0 against 25.0), which is the E29 and
E30 pattern. The perturbed column is the one selection uses.
