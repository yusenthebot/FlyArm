# B1a: whole-connectome rate controller

B1a replaces the frozen 256-neuron MaleCNS subgraph with the complete annotated MaleCNS v1.0 connectome.
Robot state input, action space, teachers, behavior cloning, DAgger, MuJoCo tasks, episode seeds, graph recipe, update rule and evaluation code are unchanged.
Spiking (LIF) dynamics are a later step and are not part of B1a.

Everything that differs from the 256-node pick-and-place baseline, stated explicitly:

| | 256-node baseline | B1a |
|---|---|---|
| Graph | 256 neurons / 4,678 edges | 166,700 neurons / 10,520,377 edges |
| Input neurons | 23 ascending | all 1,846 ascending |
| Output neurons | 43 descending | all 1,314 descending + 708 VNC motor |
| Trainable parameters (pick-and-place) | 1,050 | 78,240 |
| Readout | output state after the last internal update | mean over the 3 internal updates (window pooling) |
| Truncated BPTT window | 40 control steps | 8 control steps |
| Epochs (BC / each DAgger round) | 30 / 15 | 20 / 10, first 2 BC epochs decoder-only |
| Framework | PyTorch CPU | MLX on the Apple GPU |

The interface and parameter changes follow from using the whole CNS; the readout and training-budget changes follow the B1a plan and the cost of backpropagating through 166,700 neurons.
The GRU control is re-matched to the new parameter budget, so every model family sees the same data and budget.

## Structure

```
robot state (reach 20D / pick-and-place 37D)
    -> encoder W_in (trainable), writes ascending_neuron rows only
complete MaleCNS v1.0: 166,700 neurons / 10,520,377 directed edges / 104.4M synapses
    -> frozen signed sparse recurrence, 3 updates per 20 Hz control step
descending_neuron + vnc_motor activity, mean over the 3 updates
    -> decoder W_out (trainable)
action (reach [dx, dy, dz]; pick-and-place [dx, dy, dz, gripper])
    -> unchanged DLS IK, Menagerie Panda servos, MuJoCo contact physics
```

Constraints, each enforced in code and covered by tests:

- Observations enter only the 1,846 neurons annotated `ascending_neuron`.
- The decoder reads only the 1,314 `descending_neuron` and 708 `vnc_motor` neurons.
- The two sets are disjoint; there is no observation-to-decoder or encoder-to-decoder path.
- The connectome is frozen: it is not a module parameter, and its gradient is never formed.
- Neural state persists across control steps and is cleared only at episode reset.

With every edge removed, the outputs are exactly zero for any input, so the full connectome is the only channel between perception and action.

## Pack (`flyarm whole-brain compile`)

`whole_brain/compiler.py` builds a memory-mapped CSR pack from the three hash-pinned official exports.
The recipe is the 256-node recipe applied to the whole CNS: annotated non-glia neurons, summed contacts of at least 3 between them, no self loops, ACh +1, GABA and glutamate -1, every other transmitter 0, and incoming absolute strength normalized to at most 1 per target.

| Quantity | Value |
|---|---:|
| Annotated neurons | 166,700 |
| Neuron pairs with any synapse | 25,582,837 |
| Pairs kept (at least 3 contacts) | 10,520,377 |
| Synapses between annotated neurons | 124,177,143 |
| Synapses kept | 104,351,893 (84%) |
| Zero-sign neurons (histamine, dopamine, unclear...) | 11,609 |
| Pack size on disk | 89 MB |
| Content fingerprint | `7aa88acd850bd908317d205edfaf53bbb65e17438d98112cca8722a521c172e6` |

The "125M synapses" figure counts synapses; the matrix has one entry per connected neuron pair, so the sparse matrix has 10.5M entries, not 125M.
Rows are postsynaptic targets and columns presynaptic sources; indices are `int32`, contacts `int32`, signs `int8`.
Recompiling produces the identical fingerprint, and a run refuses any pack whose schema, recipe, source hashes, size or fingerprint differ.
All 4,678 edges of the 256-node baseline are present in the full pack with identical contacts and signs.

## Interface (`whole_brain/interface.py`)

Input and output sets are generated from the official `superclass` annotation, not hand-picked, and saved with their Body IDs, types, graph fingerprint and interface fingerprint `821bcd0d...6d4e`.

| | Neurons | Distinct types |
|---|---:|---:|
| Inputs: `ascending_neuron` | 1,846 | 567 |
| Outputs: `descending_neuron` | 1,314 | 622 (with motor) |
| Outputs: `vnc_motor` | 708 | |

Directed topology between the sets:

- 2,021 of 2,022 outputs are reachable from the inputs; hop counts are 1,901 at one hop, 118 at two and 2 at three.
- All 1,846 inputs reach some output (1,784 in one hop).
- Inputs reach 164,452 of the 166,700 neurons within six hops.
- 38,772 edges run directly from an ascending neuron to an output neuron.

The last point is the main caveat of B1a.
In the rate model about 94% of the output activity driven by a unit input comes through those direct synapses (output RMS 0.0274 with only direct edges versus 0.0293 with the full graph).
A successful whole-brain policy could therefore be a mostly VNC-local shortcut.
Every trained measured-graph policy is evaluated with a `direct_only` ablation for exactly this reason.

## Dynamics and backend

`h <- 0.5 h + 0.5 tanh(I + 0.8 W h)`, with the same constants as the 256-node policy.
Because every row of `W` has absolute sum at most 1, the update is a contraction and cannot diverge.

`whole_brain/backend_mlx.py` keeps the state `[166,700 x batch]` and the matrix in unified memory.
`W h` is a Metal kernel over the target-major CSR, one thread per target neuron and batch column with a fixed summation order, so results are bitwise deterministic.
Its vector-Jacobian product is the same kernel on the transposed CSR, so backpropagation never forms a gradient for the 10.5M edges.
Only the pooled output activity (and the action) crosses to the host each control step.
Weights are `float32`: `float16` was measured at 1.39 ms versus 1.61 ms per product with 1000x larger error, not worth the change at this size.

The backend implements the `BrainBackend` protocol (`reset`, `step`, `read_nodes`) in `whole_brain/backend.py`.
The 256-node graph runs through the same backend (`ConnectomePack.from_graph`) and reproduces the torch subgraph policy to 2e-6.

### Reuse of drosophila-brain-mlx

[drosophila-brain-mlx](https://github.com/Kisame76/drosophila-brain-mlx) (MIT) ports the Shiu et al. LIF model to MLX/Metal for FlyWire and MaleCNS.
It is the right starting point for the later `FullLIFBackend`, but it cannot serve B1a: it simulates LIF only, runs whole experiments without a stateful step API, has no autograd, and its kernel skips non-spiking neurons, which does not apply to a dense rate state.
B1a therefore only adds a single CSR product kernel and its adjoint; it does not reimplement a brain simulator.

## Go/no-go (`flyarm whole-brain check`)

Measured on an Apple M5 Max (64 GB), MLX 0.32.2, before any robot training.
All seven checks passed.

| Check | Result |
|---|---|
| Pack loads with pinned provenance | 0.07 s (memory-mapped) |
| Single-step determinism | bitwise identical across independently built backends |
| No input-output bypass | outputs exactly 0 with all edges removed and 10x inputs |
| Numerical stability | 400 control steps at 3x input: finite, max abs state 0.9998 |
| Memory | peak GPU 1.5 GiB, process RSS 1.4 GiB |
| Control rate | 4.0 ms per control step (3 full-graph updates), up to 250 Hz |
| Input reaches outputs | 99% of outputs active after 20 control steps |

Other measured costs: one truncated-BPTT chunk (batch 8, 8 control steps, forward and backward) takes 75 ms.
In closed loop with MuJoCo, an untrained full-brain policy runs a 400-step pick-and-place episode at about 220 Hz including physics.

## Reach result (`runs/whole-brain-reach-001`)

`configs/whole-brain-reach.json`: 96 teacher episodes, 16 validation, 24 held-out test targets (30000-series, identical to `reach-003`), 20 epochs with 2 decoder-only warmup epochs, truncated BPTT over 8 control steps, 3 training seeds.
The whole run took 546 s.
Teacher 24/24, zero action 0/24.

| Policy | Trainable params | Clean success per seed | 1 cm noise per seed | Mean final error, clean |
|---|---:|---|---|---:|
| Full MaleCNS | 44,835 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 4.65 mm |
| Full degree-preserving shuffle | 44,835 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 4.42 mm |
| GRU (hidden 112) | 45,139 | 24/24, 24/24, 24/24 | 24/24, 24/24, 24/24 | 5.61 mm |

Post-training checks of the three full-MaleCNS checkpoints:

| Ablation | Success per seed | Mean final error |
|---|---|---:|
| Edges off | 0/24, 0/24, 0/24 | 150.5 mm |
| Direct ascending-to-output synapses only | 24/24, 24/24, 24/24 | 5.56 mm |
| State reset every control step | 21/24, 23/24, 22/24 | 15.07 mm |

Reading, by the preregistered rules:

- **Graph-mediated control: supported.** The learned adapters are useless without the connectome (0/72).
- **Deep CNS paths: not needed for this task.** The 38,772 direct synapses alone reproduce the policy (72/72).
- **Topology advantage: none.** The shuffled full graph and the GRU are equally successful; this reach task saturates every model family, as it did at 256 nodes.

Reach therefore shows that the complete connectome can carry closed-loop control, and nothing more.
Median policy call: 4.2 ms for either full graph versus 0.38 ms for the GRU, excluding IK and physics.

## Pick-and-place result (`runs/whole-brain-pick-place-001`)

`configs/whole-brain-pick-place.json`: 96 teacher episodes, 16 validation, 24 held-out test episodes (60000-series, identical to `pick-place-observable-001`), stage-balanced behavior cloning for 20 epochs (2 decoder-only), then two DAgger rounds of 24 policy-visited episodes and 10 epochs each.
One training seed; the run took 3,795 s.
Teacher 24/24, zero action 0/24.

| Policy | Trainable params | Grasp | Lift | Stable place |
|---|---:|---:|---:|---:|
| Full MaleCNS | 78,240 | 24/24 | 18/24 | 5/24 |
| Full degree-preserving shuffle | 78,240 | 22/24 | 10/24 | 3/24 |
| GRU (hidden 143) | 78,368 | 7/24 | 7/24 | 1/24 |
| 256-node MaleCNS baseline (for reference) | 1,050 | 6/24 | 0/24 | 0/24 |

Post-training checks of the full-MaleCNS checkpoint:

| Ablation | Grasp | Lift | Stable place |
|---|---:|---:|---:|
| Edges off | 0/24 | 0/24 | 0/24 |
| Direct ascending-to-output synapses only | 15/24 | 7/24 | 2/24 |
| State reset every control step | 12/24 | 0/24 | 0/24 |

Reading, by the preregistered rules:

- **Graph-mediated control: not established.** Stable placement is 21%, far below the 70% bar, even though every grasp and most lifts succeed.
- **Unlike reach, the policy uses more than the direct synapses:** with only the 38,772 direct I/O edges it loses a third of its grasps and most lifts.
- **Temporal state matters:** clearing the neural state every step removes every lift.
- **Topology advantage: not supported.** The measured graph is ahead of the shuffle on every metric (5 vs 3 places, 18 vs 10 lifts), but with one seed and 24 episodes the difference is not significant.

Failure analysis of the 19 unsuccessful measured-graph episodes: all of them grasp, and the cube is then released 3 to 30 cm from the goal.
Transport and release accuracy, not grasping, is the bottleneck.
This is the first learned controller in the project that places the cube in more than one held-out episode.

The live UI (`flyarm whole-brain serve`) replays this checkpoint in real time and reproduces the held-out result step for step (episode 60000 placed at step 247, 1.5 cm goal error).

## Memory timescale of the rate model (2026-09-21)

Two copies of the pick-place controller that differ only in the goal inputs for 10 control steps, then receive identical inputs, converge by about a factor of five per control step.
The largest state difference falls from 2.0 to 1e-3 within 5 control steps (0.25 s) and to 1e-5 within 10 (0.5 s).
With alpha = 0.5 and row-normalized weights at gain 0.8 the dynamics are contractive, so the connectome carries information for roughly 0.1 to 0.2 s.
That is enough for the short-term state the pick-and-place skill needs (resetting the state every step removes every lift), but not for a goal that must be remembered for seconds, and a trained decoder cannot add memory the dynamics do not hold.
Working memory would need a different neuron model, for example near-critical recurrent gain or slower time constants, which is a separate question about the wiring.
