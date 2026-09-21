# Restricted-interface pick-and-place protocol

## Question

Can a controller whose observations and actions are forced through a fixed measured
MaleCNS subgraph perform a physical pick-and-place task? If it can, does its behavior
depend on graph edges, and does the measured topology outperform a degree-preserving
shuffle?

These are three separate questions. Teacher success only validates the task. A learned
MaleCNS policy must succeed before edge dependence can be interpreted. An advantage claim
requires the measured graph to beat a retrained shuffle under the same interface and data
budget.

## Restricted neural interface

- Graph: preregistered MaleCNS v1.0 256-node / 4,678-edge measured subgraph.
- Observation-write set: all 23 nodes annotated `ascending_neuron` in this graph.
- Action-read set: all 43 nodes annotated `descending_neuron` in this graph.
- Sets are disjoint and ordered by fixed Body IDs; the interface is graph-fingerprint-bound.
- Only the 37→23 encoder and 43→4 readout are trainable. The signed sparse graph is frozen.
- There is no observation-to-action, encoder-to-readout or residual bypass.
- 42/43 output nodes are reachable from the input set by a measured directed path.

“Ascending as feedback” and “descending as command” are engineering proxies. They do not
assert a natural Panda mapping in the fly.

## Physical task

The Panda and actuator model come from a clean, commit-pinned Google DeepMind MuJoCo
Menagerie checkout. A 4 cm, 25 g free-joint cube moves only through MuJoCo contact,
friction and gravity. Explicit rubber-pad collision geoms are rigid parts of the two Panda
fingers; they are not connected to the object. There is no weld/equality attachment, and
`step()` never writes object `qpos` or `mocap`.

Action at 20 Hz:

`[delta_x, delta_y, delta_z, gripper] ∈ [-1, 1]^4`

XYZ components become bounded Cartesian increments for damped-least-squares IK at fixed
orientation. Gripper `-1` closes and `+1` opens through the original finger actuator.

Privileged 37-dimensional state, all routed through the 23 declared input nodes:

- Panda q and qdot: 7 + 7
- end-effector, cube and goal XYZ: 3 + 3 + 3
- cube-minus-ee and goal-minus-cube: 3 + 3
- gripper aperture: 1
- cube linear velocity: 3
- instantaneous left/right cube contact: 2
- historical grasp/lift flags: 2

No teacher stage or teacher action is included. The historical flags make the contact task
Markov enough to distinguish pre-grasp and post-release states without revealing the
demonstrator's private state machine.

Success requires all of the following:

1. both left and right fingers contacted the cube;
2. cube was lifted at least 6 cm above its settled height;
3. final XY goal error is below 3 cm;
4. fingers are open and no longer touching the cube;
5. cube is settled on the table with speed below 0.06 m/s;
6. the release condition remains true for ten consecutive control steps.

The same four-action API teacher performs approach, descend, contact-gated close, lift,
transport, lower, open and retreat. Development seeds 0–11 must achieve at least 90%
before any learned comparison is considered valid. The current environment regression is
12/12.

## Learning and controls

Initial data are complete teacher episodes with disjoint 40000/50000/60000-series
train/validation/test seeds. Imitation loss balances the eight teacher phases so that the
short release phase is not erased by approach frames. Optional DAgger iterations run the
current policy without online correction, query teacher labels on policy-visited states,
aggregate those labeled states and retrain. During held-out evaluation, the teacher is
never called.

Matched learned controls:

1. measured MaleCNS graph;
2. retrained directed-degree-preserving shuffled graph with identical ordered I/O Body IDs;
3. exact trainable-parameter-matched MLP;
4. near-parameter-matched GRU.

Post-training checks for the measured policy:

- **edges silenced:** keep learned encoder/readout and set all recurrent graph weights to
  zero. Because I/O sets are disjoint, this structurally removes their communication path.
- **state reset every step:** preserve edges but remove temporal memory.

Every checkpoint records the graph, interface, source hashes, configuration, episode
seeds, learning curves, per-episode contact/lift/place results and causal checks. Failed
runs retain a failed status and partial artifacts; existing run directories are not
overwritten.

## Live UI evidence boundary

`flyarm serve` loads real checkpoints and advances the same environment in a background
20 Hz loop. Browser controls call server reset/run/pause/step/mode endpoints; state arrives
over WebSocket. The connectome panel uses official `somaLocation` coordinates and real
edges. The robot view is a lightweight Three.js rendering of live MuJoCo body positions;
it is not a video replay and it is not the physics engine.

The UI labels graph mediation and topology advantage independently from saved held-out
results. Missing, failed or incomplete runs remain `pending`; moving graphics cannot
promote an evidence label.
