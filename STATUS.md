# FlyArm status

Goal: a tech report whose every manipulation claim rests on three training seeds, controls run through the identical pipeline, and a protocol fixed before the results.

Mode: evolving; the rigor bar below is round one's floor.

Acceptance (report rigor, 2026-09-30):
- [ ] G1 Seeds: connectome seeds 0, 1 and 2 through the pre-registered protocol, final evaluation 32 episodes per template on every split.
- [ ] G2 Lesions: test-time silencing of central brain, optic lobes, VNC interneurons, sensory neurons, size-matched random sets, all-but-interface, and state reset; McNemar against intact.
- [ ] G3 Shuffle: degree-preserving shuffle seeds 0, 1 and 2 through the same protocol; connectome against shuffle at seed and episode level.
- [ ] G4 Report: seed table with mean and range, controls and lesion sections, statistics protocol, hyperparameters and compute, figures regenerated; claims match the evidence.

Running (see progress.md for commands):
- runs/skill-dagger-connectome-nophase-seed{1,2}, driven by scripts/protocol_pipeline.py.
- scripts/lesion_analysis.py into docs/results/manipulation-lesions.json.
- runs/queue-shuffle.sh lanes a (shuffle seeds 0 then 2) and b (seed 1 after the connectome seeds).

Gates: GRU and MLP controls are not approved; pushing needs the user's word.

Last update: 2026-09-30.
