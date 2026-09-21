# FlyArm

**用真实果蝇连接组作为固定循环网络，训练输入/输出适配器控制机械臂。**

Research MVP: a measured MaleCNS v1.0 subgraph → trainable adapters → actual Panda
MuJoCo reach and real-contact pick-and-place. This is **not a whole-brain emulation**,
not a naturally mapped fruit-fly motor system, and not a claim that biological topology
beats an MLP. See the [auditable baseline definition](docs/BASELINE.md).

## Current scope

- Pinned official MaleCNS exports; deterministic 256-neuron / 4,678-edge subgraph.
- Frozen signed sparse recurrence, trained encoder/readout; sequence behavior cloning.
- Reused Google DeepMind Menagerie Panda, MuJoCo dynamics, Gymnasium and PyTorch.
- Shared-data connectome / degree-preserving shuffled / MLP / GRU comparisons;
  teacher, zero-action and post-training edges-silenced checks.
- Three seeds; episode-disjoint splits; clean and 1 cm observation-noise evaluation.
- Real MP4 rollouts with same-frame hidden-state/action/error traces. No teacher
  corrections during learned evaluation and no direct qpos edits during stepping.
- Real-contact 4 cm cube grasping with friction, gravity, dual-finger contact, lift,
  transport, release and ten-step stable-success checks; no object weld or attachment.
- Live React/Three.js research console backed by the running MuJoCo policy: reset,
  run/pause/step, movable object/goal, selectable measured neurons and causal modes.

The task uses **privileged target position**, fixed end-effector orientation, an open
gripper and local 3D reaching. See [protocol](docs/PROTOCOL.md) for limitations.
Initial corrected reach runs solve that small task across all four model families;
this is pipeline evidence, **not** a connectome advantage. Pick-and-place is reported
separately because teacher feasibility and learned-policy success are different claims.

See [initial measured results and diagnostic history](docs/RESULTS.md) and the
[restricted pick-and-place protocol](docs/PICK_PLACE_PROTOCOL.md). The latest
pick-and-place run is deliberately reported as a negative learned-controller baseline:
the scripted teacher succeeds, while the measured-connectome policy does not yet place.

## Run locally

Tested on Apple Silicon with Python 3.12, CPU PyTorch and off-screen MuJoCo rendering.
Git and [uv](https://docs.astral.sh/uv/) are required. Downloads total about 1.1 GB
for connectome exports plus Panda assets; graph preparation benefits from ample RAM.

```bash
uv sync --frozen --python 3.12
uv run flyarm fetch
uv run flyarm prepare
uv run flyarm run --config configs/smoke.json --output runs/smoke
uv run flyarm run --config configs/reach.json --output runs/reach
uv run flyarm pick-place --config configs/pick-place.json --output runs/pick-place

cd ui && npm ci && npm run build && cd ..
uv run flyarm serve --run runs/pick-place
```

`fetch` is explicit: importing the package does not access the network. Sources are
generation-pinned, hash-checked and gitignored. `prepare` can explore other graph
sizes, but `run` intentionally accepts only the canonical 256-node MVP graph;
register a separately reviewed protocol before comparing a different graph.

`smoke` uses two epochs to test plumbing, not policy quality. Existing run directories
are never overwritten. Failures retain partial files and a `failed` results status;
retry with a **new output path**. Resume is not implemented. Training is budgeted
with a configured deadline, checked at epoch boundaries, plus bounded episode loops.

Run output includes config, exact source-file and data provenance, graph artifacts,
episode splits, demonstrations, checkpoints, validation curves, per-episode results,
MP4, PNG and JSON replay traces. Reproducibility means seed replay on the recorded
platform; cross-platform bitwise equality is not promised.

On a headless Linux machine, set `MUJOCO_GL=egl` only if a working EGL driver is
installed. This platform/rendering path has not been verified here.

Local macOS troubleshooting: if an editable install exists but `import flyarm`
fails, inspect `python -v` for `Skipping hidden .pth file`. This host sometimes
marks generated `.pth` files with `UF_HIDDEN`. A process-scoped workaround is
`PYTHONPATH=src uv run flyarm ...`; no system Python changes are required. An ordinary
non-editable wheel installation also avoids the editable `.pth` mechanism.

## Verify

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
uv run bandit -r src -q
uv run pip-audit --progress-spinner off
FLYARM_MODEL=assets/menagerie/franka_emika_panda/scene.xml uv run pytest -q
```

Without `FLYARM_MODEL`, physics integration tests are explicitly skipped; do not
confuse a unit-only pass with a verified simulation. CI fetches only the pinned
robot assets and runs physics tests; the full 1.1 GB connectome experiment is local.

## Layout and reuse

- `src/flyarm/assets.py`: pinned downloads, checksums and clean robot checkout checks.
- `graph.py`: IDs, anatomy-based selection, source binding and degree-preserving nulls.
- `models.py`: frozen graph policy and parameter-matched controls.
- `env.py`: minimal fixed-orientation reach wrapper, IK and real MuJoCo stepping.
- `experiment.py`: masked sequence imitation, held-out rollout evaluation and replays.
- `pick_place_env.py`: physical grasp/lift/place/release task and same-API teacher.
- `interfaces.py` / `pick_place_models.py`: graph-bound disjoint neural I/O and controls.
- `live.py` / `ui/`: real-time MuJoCo server and interactive 3D research console.
- `configs/`: bounded smoke and initial comparison settings.
- [架构图绘制 Prompt](docs/ARCHITECTURE_PROMPT.zh.md): ready to give another agent.
- [Third-party attribution](THIRD_PARTY.md): what is reused versus method-only references.

We do not reinvent robot meshes/physics, autodiff, GRUs or the environment API.
FlyGM / FLYNN / Shiu are method references; full-brain MLX and Stable-Baselines3 are
future options, not claimed integrations. Fly-body locomotion assets are not needed
for an arm task. Source data retains CC BY 4.0; Panda assets retain Apache 2.0;
FlyArm's own code is MIT.

## Next research milestones

1. Data-budget curves and more seeds to test sample efficiency instead of saturated success.
2. Retrained disconnected/leaky controls, graph-size ablations and OOD disturbances.
3. Improve phase-transition learning for contact/release without exposing teacher state,
   then repeat preregistered multi-seed graph-versus-shuffle comparisons.
4. Only after a measurable simulation result: visual input, larger brain models,
   reinforcement learning and separately safety-reviewed hardware trials.
