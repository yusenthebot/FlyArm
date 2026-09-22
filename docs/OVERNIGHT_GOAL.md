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

1. Data and DAgger with closed-loop phase selection (running since 05:45, expected about 09:15): runs/whole-brain-pick-place-push2-s3, -s0 and -s1 (E33); the first push runs kept the last, collapsed DAgger phase (push-s3 test 12/48 although its round-1 controller succeeded 43/48 on rollout seeds).
2. If placement stays below 90%: PPO on the motor decoder from the push checkpoint, ready as configs/ppo-pick-place-push-s0.json and -s3.json (400 iterations, nominal physics, 48 validation seeds for selection, 48 test seeds), run with `PYTHONPATH=src nohup uv run --no-sync flyarm rl ppo --config configs/ppo-pick-place-push-sN.json --output runs/ppo-pick-place-push-sN-001`; the selected (validation) checkpoint's test score is the one to report.
3. If the failure mode is grasping (cube never touched): more DAgger rounds or more demonstrations from the starts that fail.
4. Kitchen: v3 dev (front leg) and v3 body (whole-body interface) are done and negative closed loop (E30, E32); runs/flyleg-kitchen-v3-dagger-dev-001 (DAgger from perturbed starts, GRU then fly) is running; the kitchen is not on a path to 90% tonight.

## Paused on purpose (resume with kill -CONT, progress is kept)

- v2a and v2b (B1a protocol v2 topology test), paused 2026-09-22 01:25 and 01:10 to give the push the GPU.
- The dexterous run and its follow-up, the long-horizon queue and no-cue controls, the multi-task run and its launcher.
- The protocol sweep (MLP/GRU only).

## Each wake-up

1. Read this file, the "Overnight plan" and the latest entries of docs/RESEARCH_LOG.md, and `git log --oneline -10`.
2. Check the running processes and their logs; act on finished results (log, commit, next lever).
3. Keep at most one waiter per condition; keep the number of full-connectome training jobs at five or fewer.
4. Before 08:00: record labelled videos of the best new checkpoints, add dashboard featured entries with honest notes, and write the morning summary.

## Status 2026-09-22 08:32

Pick-and-place: push2 test placement 72.9% (seed 3), 64.6% (seed 0), 54.2% (seed 1); 90% not reached; PPO from the seed-3 checkpoint running (runs/ppo-pick-place-push2-s3-001).
Kitchen: every closed-loop recipe tried overnight was negative (E29, E30, E32).
Still paused: v2a, v2b, dexterous, long-horizon, multi-task, protocol sweep.
