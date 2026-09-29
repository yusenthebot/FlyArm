# Archive

Superseded code, configurations, documents and figures from earlier stages of FlyArm, kept for reference.
Nothing here is imported, tested or maintained, and paths inside these files refer to where they lived before.
The current code is in src/, the current results in the top-level README and docs/RESEARCH_LOG.md, which records every experiment in order.

- code/: the original 256-neuron-subgraph experiment line (reach and direct pick-place with the restricted interface, its torch policies and the first live UI), retired in September 2026; its source, tests and configs as they were last on main.
- configs/: configs of superseded experiment lines, flat, with their README notes: whole-brain reach smoke and pick-and-place (v2, push, seeds, shuffles) and their PPO runs, earlier kitchen imitation protocols (complete, mixed, descending, leg, v3 and other dev sweeps), kitchen scratch, warm-start, curriculum, MLP and DAPG variants off the milestone chain, the manipulation diagnostic, and the targeted and gated skill-DAgger and PPO experiments.
- docs/: planning notes and result pages of superseded setups: the 256-neuron subgraph baseline, the first pick-and-place protocol, the overnight goal of 2026-09-22, the B1a whole-connectome design note and the B2 fly-leg goal.
- figures/: the first figure set (v1/, with its sources) and old README and live-UI images that no current document uses.
- scripts/: one-off probe, analysis and queue scripts of the pick-and-place and kitchen interface studies and one manipulation study (connectome and pathway probes, input-regime, readout, linear-policy and weight-normalization probes, kitchen open-loop, perturbation and protocol sweeps, the control-feature probe, the media archiver and the shell run queues).

## Moved from

Paths cited in docs/RESEARCH_LOG.md and older commits resolve here.

| Old path | New path |
|---|---|
| docs/archive/* | archive/docs/* (docs/archive/README.md merged into this file) |
| docs/B2_FLYLEG_GOAL.md, docs/WHOLE_BRAIN.md | archive/docs/ |
| docs/figures/v1/ | archive/figures/v1/ |
| docs/images/pick-place-rollout.png | archive/figures/pick-place-rollout.png |
| docs/flyarm-live-ui-light.png, docs/flyarm-live-ui-concept-light.png | archive/figures/ |
| configs/<name>.json, configs/<name>.README.md not in configs/ | archive/configs/<name>.json, archive/configs/<name>.README.md |
| scripts/<name> not in scripts/ | archive/scripts/<name> |
