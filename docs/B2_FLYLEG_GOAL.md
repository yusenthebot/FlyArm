# B2 goal: the Franka arm as the fly's left front leg

## Goal

A frozen, complete fruit-fly CNS (MaleCNS v1.0, 166,700 neurons) controls a Franka Panda on a standard, Mac-runnable manipulation benchmark, wired to the robot only the way a fly is wired to its own front leg.
The arm's joint state enters through the fly's own left front-leg proprioceptive afferents.
Scene state enters through the fly's head sensory neurons.
Joint commands are read only from the fly's left front-leg motor neurons.
Everything in between, including every nonlinearity and all memory, is the fly's measured wiring; the only trained parts are linear maps into sensory neurons and out of motor neurons.

Success means four things, each measured, none assumed:

1. **Benchmark:** a D4RL FrankaKitchen score (complete, partial, mixed) reported against published BC numbers and against our own matched controls on identical data.
2. **Generalization:** the partial and mixed splits, which never demonstrate the full target task sequence, plus out-of-distribution perturbations (robot and object initial-state noise beyond training, observation noise).
3. **Causal dependence on the connectome:** edges off, direct sensory-to-motor synapses only, deafferented leg (proprioception silenced) and head-sensory deprivation, each evaluated on the trained policy.
4. **Biological specificity:** the measured wiring against a degree-preserving shuffle of the whole CNS with the same sensory and motor neurons, over at least three seeds.

## Why this design

"A fly brain controls a robot arm" is most convincing when the arm replaces a body part the fly already controls, through the neurons that normally serve it.
The fly's front leg is its most dexterous limb: it reaches, grooms and grasps with the tarsal claws.
MaleCNS annotates exactly the neurons needed:

| Channel | Neurons | Selection |
|---|---:|---|
| Motor output | 68 | `vnc_motor`, subclass `fl`, left side: coxa promotor/remotor/rotators, Tr flexor/extensor, Fe reductor, Ti flexor/extensor, Ta depressor/levator, ltm (claw) |
| Proprioception | 23 | `vnc_sensory`, chordotonal organ / hair plate / campaniform sensilla, entry nerve ProLN, left side |
| Exteroception | 4,868 | `cb_sensory`: antennal, olfactory, gustatory and head mechanosensory neurons |

FrankaKitchen is the benchmark that fits this interface without translation.
Its actions are joint velocities of the 7 arm joints and 2 fingers, so motor-neuron activity drives joints directly, with no inverse kinematics or Cartesian shortcut between the fly and the arm.
It is a Franka, it runs on macOS through Gymnasium-Robotics and Minari, and its datasets and scores are standard (D4RL).
MetaWorld was considered and rejected: it pins `mujoco==3.3.0`, which would change the verified FlyArm physics, and it drives a Sawyer hand by mocap rather than joints.

## Measured feasibility (before any training)

Rate dynamics as in B1a (`h <- 0.5 h + 0.5 tanh(I + 0.8 W h)`, row-normalized), 12 control steps of drive, sensitivity of left front-leg motor neurons per input neuron:

| Input channel | Hops to motor neurons | Gradient per input | Motor RMS |
|---|---|---:|---:|
| Left front-leg proprioceptors | 21 at 1, 46 at 2 | 3.2e-3 | 1.3e-3 |
| Head sensory neurons | mostly 2-3 | 3.8e-5 | 1.2e-4 |
| Photoreceptors R1-R8, recipe v1 | 3-5 | 0 | 0 |
| Photoreceptors, histamine inhibitory | 3-5 | 2.3e-8 | 9.4e-8 |

Vision is not used in B2.
Photoreceptors are histaminergic, which recipe v1 maps to sign 0, so the eye is silent; even with the biologically correct inhibitory histamine sign the visual path is attenuated by five orders of magnitude, with row-normalized or spectrally normalized weights alike.
A visual channel needs different dynamics (spiking or near-critical) and is a later milestone.
The leg reflex arc is the strongest path in the CNS, which is what makes this interface trainable.

## Milestones

1. **Benchmark adapter.** Gymnasium-Robotics FrankaKitchen and Minari D4RL kitchen datasets as a FlyArm task: offline behavior cloning data, closed-loop evaluation with the benchmark's own task-completion score, verified by replaying dataset actions.
2. **Front-leg interface.** Anatomical I/O selection generated from annotations, go/no-go checks (reachability, per-channel sensitivity, no bypass), and interface records with fingerprints.
3. **Training and baselines.** Front-leg MaleCNS, full shuffle, edges off, GRU and MLP on kitchen-complete, then partial and mixed; three seeds each.
4. **Generalization and lesions.** Noise-ratio OOD sweeps, deafferentation, head-sensory deprivation, direct-only.
5. **Live view.** Kitchen rollouts and the live UI showing leg motor-neuron and proprioceptor activity next to the arm.

## Evidence rules

- Benchmark claim: the front-leg MaleCNS score, with 95% intervals over seeds and episodes, beside published BC (complete 65.0, partial 38.0, mixed 51.5) and our same-data MLP/GRU.
  The published numbers come from the original D4RL v0 environment; Minari's v2 datasets run in Gymnasium-Robotics FrankaKitchen-v1, so they are approximate references and the same-data controls are the real comparison.
- Graph-mediated: MaleCNS score at least 50 with edges off below 5.
- Uses the CNS beyond reflexes: direct sensory-to-motor synapses alone lose at least a third of the score.
- Uses both senses: deafferentation and head-sensory deprivation each reduce the score.
- Topology advantage: measured beats shuffle by a significant margin over at least three seeds; otherwise the result is "the fly CNS is a usable controller", not "its wiring is better".

## Findings while building the benchmark

**Copycat failure of behavior cloning with velocity inputs.**
With the full 59-dimensional observation, every BC controller scored 0 on kitchen-complete, including a two-layer MLP with validation MSE 0.008.
Replaying a demonstration open loop scores 100, and replay tolerates action noise of 0.005 (4/4 tasks), so the environment and data are consistent.
The robot joint velocities in the observation are almost exactly the previous velocity command, so the policy learns to repeat its last action and never leaves the start pose.
Removing all velocities from the controller input (robot joint positions plus object joint positions, 30 dimensions) raised the same MLP from 0 to 45 on 10 episodes.
All controllers therefore see the same position-only features; recurrent controllers, including the fly CNS, can still infer velocity from their own state.
The benchmark itself is unchanged: its environment, observations, actions and scoring are used as released.

**Motor-neuron readout scale.**
Left front-leg motor-neuron activity under behavior-cloning inputs has a median RMS of 2.8e-4 (range 4.4e-5 to 6.4e-3), far below the unit-scale inputs the linear decoder expects, and Adam could not grow the decoder weights fast enough (validation MSE 0.074 after 30 epochs).
Before training, each output neuron is now scaled by the inverse of its RMS activity on the training episodes (a frozen buffer saved with the checkpoint); validation MSE then reached 0.027 after 20 epochs.
The rescaling touches only the declared motor neurons and adds no pathway.

**Compatibility.**
Gymnasium-Robotics 1.4.2 cannot construct FrankaKitchen with MuJoCo 3.13 because it tests joint types with `in (enum, ...)`, which the newer bindings evaluate as false for numpy integers.
`flyarm.benchmarks._robotics_compat` replaces its four joint accessors with integer comparisons; MuJoCo is not downgraded.

## Live UI

```bash
uv run flyarm flyleg serve --run runs/flyleg-kitchen-complete-001 --seed 0 --port 8770
```

The kitchen scene, every visible geom of FrankaKitchen-v1, is exported from the compiled MuJoCo model and posed from live body states; the complete CNS is drawn with the three interface groups in their own colors.
Sensory afferents have no soma inside the CNS; the 4,676 that have drawn partners are placed at the contact-weighted centroid of their postsynaptic partners' somata and labelled as such.
Modes: measured CNS, shuffled CNS (trained separately), edges off, direct sensory-to-motor synapses only, deafferented leg, head senses removed.

## Overnight results (2026-09-21)

Full tables: [overnight report](results/overnight-2026-09-21.md), generated by `flyarm report` from the run artifacts.

FrankaKitchen-complete, 3 training seeds x 40 episodes, D4RL normalized score:

| Controller | Score | Notes |
|---|---:|---|
| Fly CNS, leg proprioceptors + head senses | 0.0 | removing head senses gave 25 on seed 0 |
| Shuffled CNS, same interface | 0.0 | |
| MLP behavior cloning | 8.3 ± 14.4 | microwave on seed 2 only |
| GRU, parameter-matched | 0.0 | |
| Fly CNS, leg proprioceptors only | 7.3 ± 12.1 | seed 2: slide cabinet in 34/40 episodes |
| Shuffled CNS, proprioceptors only | 0.0 | |

FrankaKitchen-mixed (1 seed, 15 epochs): all four controllers scored 0.
No controller learned the benchmark under this behavior-cloning budget, so the benchmark does not yet discriminate between them.
A constant action (the edges-off lesion) scored 25 on one seed by flipping the light switch, so single-task completions are only meaningful with lesions.
The one lesion-validated skill is the proprioception-only fly leg on seed 2: edges off, direct synapses only, deafferentation and state reset each reduce its 34/40 slide-cabinet openings to 0, and the matched shuffle scores 0.

B1a pick and place with 3 seeds (72 held-out episodes per controller): lift 58/72 for the complete MaleCNS, 27/72 for the shuffled CNS and 33/72 for the GRU, consistent in every seed (paired episodes 36 wins, 5 losses against the shuffle, exact p = 4e-7; seed-level n = 3).
Stable placement: 14/72, 10/72 and 4/72.
