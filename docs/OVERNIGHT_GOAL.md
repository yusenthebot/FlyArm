# Overnight goal, 2026-09-22

User goal (verbatim): "把果蝇大脑做到90以上成功率，复杂task".
The user is away overnight and checks in the morning.
The fly brain stays the protagonist: the controller is W_in -> frozen complete MaleCNS -> W_out -> tanh, and only the linear maps are trained.

## Acceptance

- Primary: the fly controller places the cube in at least 90% of held-out B1a pick-and-place test episodes (48 test seeds, 60000 to 60047), stable place as defined by the environment, with the checkpoint chosen on validation seeds only.
- Secondary: the kitchen fly controller improves under the closed-loop protocol v3 (perturbed starts, research log E30), measured against the replay floor (16 to 19) and the tracker ceiling (100).
- Already above 90%: the dexterous hand, connectome seed 0, 62/64 in-hand rotations (paused by the user; do not resume without asking).
- Every result goes into docs/RESEARCH_LOG.md (question, config, run, counts, reading, negatives included) and is committed locally; never push.

## Levers, in order

1. Data and DAgger (running): runs/whole-brain-pick-place-push-s0 and -push-s3, 384 demonstrations, 3 DAgger rounds, resync teacher (E31).
2. If placement stays below 90%: PPO on the motor decoder from the push checkpoint (flyarm rl ppo, config like configs/ppo-pick-place.json with base_run set to the push run, nominal physics, validation-seed selection), then evaluate the selected checkpoint on the 48 test seeds.
3. If the failure mode is grasping (cube never touched): more DAgger rounds or more demonstrations from the starts that fail.
4. Kitchen: runs/flyleg-kitchen-v3-dev-001 (front leg, fly and GRU) and runs/flyleg-kitchen-v3-body-dev-001 (whole-body interface, E32 probe: decoder-only fit 0.093 against 0.111).

## Paused on purpose (resume with kill -CONT, progress is kept)

- v2a and v2b (B1a protocol v2 topology test), paused 2026-09-22 01:25 and 01:10 to give the push the GPU.
- The dexterous run and its follow-up, the long-horizon queue and no-cue controls, the multi-task run and its launcher.
- The protocol sweep (MLP/GRU only).

## Each wake-up

1. Read this file, the "Overnight plan" and the latest entries of docs/RESEARCH_LOG.md, and `git log --oneline -10`.
2. Check the running processes and their logs; act on finished results (log, commit, next lever).
3. Keep at most one waiter per condition; keep the number of full-connectome training jobs at five or fewer.
4. Before 08:00: record labelled videos of the best new checkpoints, add dashboard featured entries with honest notes, and write the morning summary.
