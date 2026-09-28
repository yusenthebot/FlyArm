# FlyArm

**一只果蝇的完整大脑接线图，不改一根线，驱动一台 Franka 机械臂。**

A frozen, complete fruit-fly central nervous system (MaleCNS v1.0: 166,700 neurons, 10.5 M connections) is the controller of a simulated Franka arm.
Only a linear map into sensory neurons and a linear map out of motor neurons are trained.
It runs in real time on a Mac (MLX, 4 ms per control step).

**Videos: the frozen fly connectome controls the arm**
- [Drawers, a lidded cabinet, shelves, a bin and stacking](docs/videos/fly-brain-highlights.mp4): one successful test episode of each of the seven task templates (2 minutes).
- [Objects never seen in training](docs/videos/fly-brain-unseen-objects.mp4): the same seven templates with held-out scanned objects (2 minutes).
- [Task orders never seen in training, 5 to 8 subgoals](docs/videos/fly-brain-unseen-composition.mp4): three of the four held-out compositions, including the 8-subgoal full cleanup (82 seconds).

![Kitchen live view: the complete CNS, the Franka as the fly's left front leg, and the motor plan decoded from its motor neurons](docs/images/ui-kitchen.png)

## How it works

![Control network: trained linear maps write the observation into sensory neurons, the frozen MaleCNS runs three rate-dynamics steps per control step, and trained linear maps read motor neurons into actions](docs/figures/control_network-figure.png)

- **The brain is not trained.** Connection weights come from measured synapse counts, and the only path from inputs to outputs runs through the connectome.
- **Biologically mapped I/O.** In FrankaKitchen the arm is the fly's left front leg: joint angles enter through its 23 leg proprioceptors, the scene through 4,868 head sensory neurons, and actions are read from its 68 leg motor neurons, all selected by annotation rules.
- **Causal controls, as in neuroscience.** Every result is compared with a degree-preserving shuffle of the same CNS, a parameter-matched GRU and an MLP, and checked with lesions: edges off, direct synapses only, deafferentation, state reset every step.

## Milestone: FrankaKitchen solved by the frozen connectome

With only the two linear maps trained (imitation, then PPO with a demonstration term, an ordered completion bonus and potential-based shaping), the Franka completes all four kitchen-complete tasks in every test episode of the official environment:

| Start (50 test episodes each) | Benchmark score | All four tasks | Strict score, motion-quality controller |
|---|---:|---:|---:|
| Benchmark start | 100.0 | 50/50 | 69.5 |
| 0.2 rad perturbed (never trained on) | 100.0 | 50/50 | 70.5 |
| 0.3 rad perturbed (never trained on) | 100.0 | 50/50 | 68.0 |

Replicated on three PPO seeds; controls for the RL stage are still to run, so a wiring advantage is not claimed.
Details, recipe, videos and open items: [kitchen milestone](docs/MILESTONE_KITCHEN.md).
Demo: [four episodes from 0.3 rad perturbed starts](docs/videos/kitchen-four-tasks-perturbed.mp4).
Pick and place after PPO on the output map: 95.8%, 97.9% and 91.7% placement over three seeds (48 test episodes each).

## Articulated manipulation (in progress)

Drawers, a lidded cabinet, a bin and stacking with 41 scanned household objects, tasks of 3 to 8 subgoals (1,100 to 2,600 control steps), five preregistered splits ([environment](docs/MANIPULATION_ENV.md)).
Frozen connectome with a trained nonlinear sensory encoder and a linear readout, skill-level DAgger with an observable motor-phase cue, then PPO; full-task success from true starts:

| Split | Imitation | After PPO |
|---|---:|---:|
| iid | 51.8% | 67.9% |
| unseen objects | 57.1% | 75.0% |
| unseen furniture | 28.6% | 39.3% |
| unseen composition | 21.9% | 43.8% |

Highlights: [one successful test episode per task template](docs/videos/fly-brain-highlights.mp4) (checkpoint policy-0300), [unseen objects](docs/videos/fly-brain-unseen-objects.mp4) and [unseen task orders](docs/videos/fly-brain-unseen-composition.mp4) (PPO with the encoder trained too, iteration 150 of runs/ppo-manipulation-encoder-002; research log E61).
Demo: [one success and one failure per task template, iid split](docs/videos/manipulation-connectome-ppo-iid.mp4) (checkpoint policy-0300; in 4 of 7 templates all four candidate episodes succeeded, so no failure is shown).
Controls through the same pipeline (imitation only, one seed): matched MLP 26.8% iid, matched GRU 66.1% iid, degree-preserving shuffle of the connectome 44.6% iid against the measured connectome's 51.8%; further controls wait until the connectome result is final.
Why earlier attempts failed and what changed: [architecture analysis](docs/ARCHITECTURE_ANALYSIS.md).

## Earlier results

Real-contact pick and place, 3 training seeds x 24 held-out episodes:

| Controller | Grasp | Lift | Stable place |
|---|---:|---:|---:|
| Complete MaleCNS | 69/72 | **58/72** | 14/72 |
| Shuffled CNS (same degrees) | 65/72 | 27/72 | 10/72 |
| GRU, parameter-matched | 43/72 | 33/72 | 4/72 |
| MaleCNS, every edge removed | 11/72 | 0/72 | 0/72 |
| MaleCNS, state reset every step | 60/72 | 0/72 | 0/72 |

![Complete MaleCNS rollout: approach, grasp, lift, place](docs/images/pick-place-rollout.png)

In progress: a replication with 6 seeds and several shuffles per seed, and D4RL FrankaKitchen with ACT-style action chunking for every controller.
The first new seed favours the shuffle, so a topology advantage is not yet a claim.
Details: [whole-brain controller](docs/WHOLE_BRAIN.md), [kitchen protocol and findings](docs/B2_FLYLEG_GOAL.md).

![Pick-and-place live view](docs/images/ui-pick-place.png)

## Run

Apple Silicon, Python 3.12 and [uv](https://docs.astral.sh/uv/); downloads are about 1.1 GB.

```bash
uv sync
uv run flyarm fetch
uv run flyarm whole-brain compile
uv run flyarm whole-brain run --config configs/whole-brain-pick-place.json --output runs/pick-place
uv run flyarm flyleg run --config configs/flyleg-kitchen-complete-chunk.json --output runs/kitchen

cd ui && npm ci && npm run build && cd ..
uv run flyarm flyleg serve --run runs/kitchen --seed 0 --port 8771
```

## Scope

- An abstract rate model on the measured wiring, not a physiological or spiking emulation of the fly.
- Observations are simulator state, not vision.
- Simulation only.

Data: MaleCNS v1.0 (CC BY 4.0).
Robot: MuJoCo Menagerie Franka Panda (Apache 2.0) and Gymnasium-Robotics FrankaKitchen.
Code: MIT; see [third-party attribution](THIRD_PARTY.md).
