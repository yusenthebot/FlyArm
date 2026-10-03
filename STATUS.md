# FlyArm status

Mode: evolving. Two goals share the one GPU; the connectome stays frozen in both.

## Goal 1: report rigor (since 2026-09-30)
Every manipulation claim in the report rests on three training seeds, controls run through the identical pipeline, and a protocol fixed before the results.
- [ ] G1 Seeds: connectome seeds 0, 1 and 2 through the pre-registered protocol, final evaluation 32 episodes per template on every split.
- [x] G2 Lesions (seed 0 done; repeat on seeds 1 and 2): state reset, anatomical groups against size-matched random sets, all-but-interface; McNemar against intact.
- [ ] G3 Shuffle: degree-preserving shuffle seeds 0, 1 and 2 through the same protocol; connectome against shuffle at seed and episode level.
- [ ] G4 Report: seed table with mean and range, controls and lesion sections, statistics protocol, hyperparameters and compute; claims match the evidence.

## Goal 2: generalization (since 2026-10-03)
Close the gap on held-out furniture (36.6%) and held-out task compositions (26.6%, retrieve_to_shelf 0%) without losing iid (76.3%).
- [ ] G5 Diagnose with evidence: out-of-range inputs (scripts/ood_clamp_probe.py), where unseen-furniture episodes stop, why retrieve_to_shelf never opens its drawer.
- [ ] G6 Intervene: training-side changes only (data, observation frame, curriculum, augmentation); compare on dev seeds of the held-out splits (declared in the log before use, disjoint from the final seeds 500 onward).
- [ ] G7 Targets on fresh final episodes: unseen furniture >= 50%, held-out compositions >= 35%, iid within its interval; then replicate the recipe on three seeds.

Running:
- Connectome seeds 1 and 2: imitation resumes after round 6 (drivers scripts/protocol_pipeline.py).
- Shuffle seed 2 (lane a), then seed 0 resumed (runs/queue-shuffle-seed0-resume.sh); seed 1 after the connectome seeds (lane b).
- scripts/ood_clamp_probe.py into docs/results/ood-clamp-probe.json.

Gates: GRU and MLP controls not approved; pushing needs the user's word; LIBERO or new dependencies need a yes.

Last update: 2026-10-03.
