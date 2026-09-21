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
