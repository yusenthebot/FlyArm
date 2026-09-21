# Initial local experiment: 2026-09-20

**Outcome: the pipeline works, but this reaching task provides no evidence that
the fruit-fly topology improves control.** Every trained family saturates success;
even silencing the trained connectome's recurrent edges preserves success here.

Run: `runs/reach-003`, using `configs/reach.json`. Three training seeds × 24 shared
test targets, 96 training episodes, 16 validation episodes, 35 epochs. Test seeds
30000–30023. These are 72 rollouts per family, **not 72 independent target samples**.

| Policy | Trainable parameters | Clean success per seed | 1 cm noise success per seed | Mean final error, clean |
|---|---:|---|---|---:|
| True connectome | 6,147 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 5.37 mm |
| Degree-preserving shuffled | 6,147 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 4.89 mm |
| MLP | 6,147 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 5.54 mm |
| GRU (hidden=35) | 6,093 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 6.38 mm |

Teacher: 24/24. Zero action: 0/24. Post-training edges silenced: 24/24 for each of
three connectome checkpoints. The input/readout adapters and residual temporal leak
remain in that ablation; it does not establish that all recurrence is unnecessary.
No statistically significant advantage is claimed from the small error differences.

On the recorded Apple Silicon host, median per-episode median policy-call times:
connectome 0.041 ms, shuffled 0.041 ms, MLP 0.013 ms, GRU 0.016 ms. These exclude IK,
physics and rendering, and are **not** whole-brain or hardware latency benchmarks.
The full bounded run took about 20.2 s after asset preparation. Timing can vary.

## Provenance and replay

- Graph content SHA256: `7a5018c5481d14307e1efec305f4f448648545e5feda7caf9297dab518f4da5d`.
- Schema 1, recipe `descending-contact-v1`; 256 nodes, 4,678 directed edges.
- Every input export was verified against pinned SHA256 and GCS generation.
- [Provenance](results/reach-003-provenance.json) includes exact package/platform
  versions, source-file hashes and the explicit RNG plan.
- [Per-episode results](results/reach-003.json) retain clean/noisy outcomes and
  the trained-edges-silenced check, not only averages.
- Full local checkpoints, demonstrations and six-episode MP4/state trace are under
  the gitignored run directory; regenerate with the README commands.
- The interactive preview uses actual recorded h states. Its 3D sphere layout is
  schematic and only displays the strongest 200 edges for legibility. It is neither
  anatomical coordinates nor physiological membrane voltage. The controller itself
  uses all 4,678 selected edges.

## Retained diagnostic history

`reach-001` had a real controller defect: `mju_subQuat` returned a local-frame rotation
error that was passed directly into a world-frame rotational Jacobian. Teacher and
most policies succeeded at only 8/24 diagnostic targets; this run is **not valid
model-comparison evidence**. The fix rotates the error into the world frame and adds
a regression test on all 24 development targets. First-frame target rendering was
also fixed with `mj_forward` after mocap updates.

Those 20000-series targets became development/regression data. The corrected
comparison uses fresh 30000-series targets. `reach-002` checked the controller fix;
`reach-003` repeated it with hardened graph provenance and source/seed logging.
The original local artifacts have not been deleted or overwritten.

## Verification

19 tests passed with actual MuJoCo assets, including all 24 controller-regression
targets. Ruff, formatting, mypy and Bandit pass. `pip-audit` found no known dependency
vulnerabilities; the unpublished local FlyArm package itself is not in PyPI's audit
database. Independent Python review found no remaining blocker after provenance and
failed-run reporting fixes. Remote CI status should be checked separately.

Next useful experiment: predefine a training-data budget sweep and stronger
perturbations/OOD targets, with matched controls and additional seeds. Simply running
longer on this saturated target distribution cannot demonstrate a connectome benefit.
