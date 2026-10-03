# FlyArm progress

## Current state (2026-09-30)
- Reported controller: runs/ppo-manipulation-nophase-001 iteration 500 (no phase cue), final evaluation iid 76.3%, unseen objects 82.6%, unseen furniture 36.6%, held-out compositions 26.6%; one seed.
- Protocol for seeds and the shuffle is in docs/RESEARCH_LOG.md, "Report rigor plan 2026-09-30".
- Lesions (E63, seed 0): state reset 0/112; VNC interneurons 46 vs random 84/79 (specific); optic lobes 94 vs random 39/1 (dispensable); central brain and sensory like random; all-but-interface 16/112.
- Path analysis (docs/results/connectome-paths.json): 99% of descending and 84% of motor neurons are one synapse from an input, but 69% and 85% of their incoming weight comes from interior neurons.

## Paused 2026-10-03 (user: 暂停训练)
- SIGSTOP on connectome seeds 1 and 2 PPO (no time budget; at iteration < 50, safe to hold) and the detour imitation run (round 8 in progress; its 48 h budget from 2026-10-03 03:08 ends 2026-10-05 03:08, after which it would stop with TimeoutError and resume from round 7). Resume: `kill -CONT $(pgrep -f 'skill-dagger|rl manipulation|protocol_pipeline')`.

## Commands
- One seed end to end: `PYTHONPATH=src:scripts .venv/bin/python scripts/protocol_pipeline.py --kind connectome|shuffled --seed N` (resumable; results in docs/results/protocol/).
- Lesions: `PYTHONPATH=src:scripts .venv/bin/python scripts/lesion_analysis.py --policy ppo:runs/ppo-manipulation-nophase-001@500 --output docs/results/manipulation-lesions.json`.
- Seed 0 under the protocol: runs/queue-seed0-protocol.sh scores round 9 and iteration 300 into docs/results/manipulation-selection.json.

## What worked
- LesionedDynamics wraps the frozen dynamics and reuses BrainPolicy.with_dynamics; unit tests check the empty lesion equals intact.

## What did NOT work
- Pausing with SIGSTOP for days: skill-DAgger's max_seconds deadline (time.monotonic) kept running, so all three imitation runs raised TimeoutError on resume (2026-10-03); completed rounds were kept and the drivers resumed. For a pause longer than a few hours, stop the processes instead and resume from the last round.
- The venv's .pth files carried the macOS hidden flag (iCloud), which Python 3.12 skips, so `flyarm` could not import; cleared with chflags nohidden; launch with PYTHONPATH=src as a guard.
- pytest without FLYARM_MODEL skips 65 simulator tests; always set FLYARM_MODEL to the Panda scene.xml for the local CI run.

## Next
- When the detour run finishes: select among rounds 7 to 9 on validation, compare with seed 0 round 8 on development seeds (iid, unseen furniture, unseen composition); if better, PPO and final evaluation.
- Repeat lesions on seeds 1 and 2 when their PPO checkpoints are selected.
- When seeds finish: seed table and statistics; then shuffle comparison.

## Frontier
- Ceiling now: state-based observation with a task cue from the plan; one seed.
- Next frontiers: goal-only task cue (the controller infers the order), perception by privileged-teacher-to-student distillation from rendered images, a public benchmark (LIBERO needs a dependency: ask).
- Escalation ladder: seeds and controls, then weaker cue, then vision, then public benchmark.
