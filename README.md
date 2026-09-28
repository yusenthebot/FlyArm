# FlyArm

**一只果蝇的完整大脑接线图，不改一根线，驱动一台 Franka 机械臂。**

A frozen, complete fruit-fly central nervous system (MaleCNS v1.0: 166,700 neurons, 10.5 M measured connections) is the controller of a simulated Franka arm.
The connectome is never trained; only a sensory encoder into its ascending neurons and a linear decoder out of its descending and motor neurons are.
It runs in real time on a Mac (MLX, about 4 ms per control step).

**Videos: the frozen fly connectome controls the arm**
- [Drawers, a lidded cabinet, shelves, a bin and stacking](docs/videos/fly-brain-highlights.mp4): one successful test episode of each of the seven task templates (2 minutes).
- [Objects never seen in training](docs/videos/fly-brain-unseen-objects.mp4): the seven templates with held-out scanned objects (2 minutes).
- [Task orders never seen in training](docs/videos/fly-brain-unseen-composition.mp4): held-out compositions of 5 to 8 subgoals, including the 8-subgoal full cleanup (80 seconds).
- [FrankaKitchen](docs/videos/kitchen-four-tasks-perturbed.mp4): all four tasks from perturbed starts never trained on.

## Pipeline

![FlyArm pipeline: observation, trained encoder, frozen MaleCNS connectome, trained motor decoder, MuJoCo, in closed loop; training by imitation then PPO with a demonstration term](docs/figures/pipeline-figure.png)

- **Input.** Simulator state: the arm, the objects, the scene and, on the manipulation benchmark, a task cue (current skill, target and observable motor phase).
- **The brain is not trained.** Weights are measured synapse counts; each control step runs three steps of a rate model on the whole CNS, and the only path from inputs to outputs runs through it.
- **Trained interface.** An encoder writes currents into the 1,846 ascending neurons; a linear decoder reads the 1,314 descending and 708 VNC motor neurons into the action.
- **Training.** Imitation of a teacher with DAgger, then PPO with a demonstration term (DAPG); on the manipulation benchmark PPO also trains the encoder, through one control step of the frozen connectome.

Figure caption and sources: [docs/figures/pipeline-caption.md](docs/figures/pipeline-caption.md).

## Results

### Articulated multi-step manipulation

Drawers, a lidded cabinet with a shelf, a bin, a target region and stacking, with 41 scanned household objects; seven task templates of 3 to 7 subgoals and four held-out compositions of 5 to 8 (1,100 to 2,600 control steps at 20 Hz); five preregistered splits ([environment](docs/MANIPULATION_ENV.md)).
Full-task success from true starts, 32 fresh test episodes per template never used for training or checkpoint selection, with Wilson 95% intervals ([results](docs/results/manipulation-final.json)):

| Split | Imitation | + PPO (decoder) | + PPO (encoder and decoder) |
|---|---:|---:|---:|
| iid | 51.3% | 56.3% | **69.6%** (63 to 75) |
| unseen objects | 49.6% | 53.1% | **75.0%** (69 to 80) |
| unseen furniture | 27.2% | 32.1% | **36.6%** (31 to 43) |
| unseen composition | 18.8% | 25.0% | **26.6%** (20 to 35) |

The privileged scripted teacher solves 111 of the first 112 of these iid episodes (and 111, 110 and 61 of 112, 112 and 64 on the held-out splits).
Failures concentrate in placing on the shelf and in the region, in furniture outside the training ranges, and in retrieving from a drawer to the shelf.

### FrankaKitchen

All four tasks of D4RL kitchen-complete in the official Gymnasium-Robotics environment, 50 test episodes per condition ([milestone](docs/MILESTONE_KITCHEN.md)):

| Start | Benchmark score | All four tasks | Strict score, motion-quality controller |
|---|---:|---:|---:|
| Benchmark start | 100.0 | 50/50 | 69.5 |
| 0.2 rad perturbed (never trained on) | 100.0 | 50/50 | 70.5 |
| 0.3 rad perturbed (never trained on) | 100.0 | 50/50 | 68.0 |

### Controls

Whether the measured wiring matters is not yet claimed.
A degree-preserving shuffle of the connectome, a parameter-matched GRU and an MLP run through the same pipeline; so far only their imitation stage exists on the manipulation benchmark (iid: measured connectome 51.8%, shuffle 44.6%, GRU 66.1%, MLP 26.8%, one seed each), and the full comparison follows once the connectome result is final.

## Scope and limits

- Observations are simulator state, not vision, and the manipulation task cue (skill and motor phase) comes from the task plan and from geometric predicates of the simulator state.
- An abstract rate model on the measured wiring, not a physiological or spiking emulation of the fly.
- Simulation only; the manipulation results are one training seed so far (the kitchen four-task stage replicates on three PPO seeds).

## Run

Apple Silicon, Python 3.12 and [uv](https://docs.astral.sh/uv/); the connectome download is about 1.1 GB.

```bash
uv sync
uv run flyarm fetch
uv run flyarm whole-brain compile

# manipulation: skill-level DAgger, PPO on the decoder, then PPO on encoder and decoder
uv run flyarm manipulation skill-dagger --config configs/skill-dagger-connectome.json --output runs/skill-dagger-connectome-002
uv run flyarm rl manipulation --config configs/ppo-manipulation-skill-dagger.json --output runs/ppo-manipulation-skill-dagger-002
uv run flyarm rl manipulation --config configs/ppo-manipulation-encoder.json --output runs/ppo-manipulation-encoder-002
uv run flyarm rl manipulation --config configs/ppo-manipulation-encoder-clipped.json --output runs/ppo-manipulation-encoder-003
uv run python scripts/final_evaluation.py --policy final=ppo:runs/ppo-manipulation-encoder-003@150 --output runs/final.json

# kitchen: see docs/MILESTONE_KITCHEN.md for the full recipe
uv run python scripts/kitchen_final_eval.py runs/kitchen.json LABEL=RUN/policy-XXXX.safetensors
```

Each config names the run it starts from, so the stages run in this order.
The history of every experiment, including what did not work, is in [docs/RESEARCH_LOG.md](docs/RESEARCH_LOG.md).

Data: MaleCNS v1.0 (CC BY 4.0).
Robot: MuJoCo Menagerie Franka Panda (Apache 2.0) and Gymnasium-Robotics FrankaKitchen.
Code: MIT; see [third-party attribution](THIRD_PARTY.md).
