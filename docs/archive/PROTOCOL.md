# MVP protocol

## Question and evidence boundary

Does a measured biological connectivity prior help a trained arm-control adapter?
This project currently establishes a reproducible experiment, **not** an advantage.
The selected subgraph is not a whole-brain simulation, and leaky tanh states are not
biophysically fitted membrane voltages. The task is privileged-state local reaching,
not vision-based manipulation. There is no hardware integration.

## Data and implementation

- MaleCNS v1.0 official feather exports, immutable GCS generations and pinned SHA256.
- Original neuron identities remain int64. Unknown/modulatory NT signals get zero
  recurrent weight. ACh excitatory and GABA/glutamate inhibitory is a simplified
  source-sign assumption, not a receptor-specific biophysical assertion.
- Descending cell-type seeds: DNa02, DNg13, DNge104, DNp01. Grow breadthwise using
  summed incoming/outgoing contacts; integer IDs break score ties. No task reward,
  evaluation performance, or learned weights are involved in graph selection.
- Exclude glia/unclassified superclasses, self-loops, edges with fewer than 3 contacts.
- MVP instance: 256 nodes, 4,678 directed edges. These are a small fraction of the
  source connectome; graph metadata records the selection denominator and coverage.
- Fixed sparse adjacency A[post, pre], source-signed contacts, destination incoming
  absolute-sum normalization. h = .5h + .5 tanh(encoder(x) + .8Ah).
- Only encoder/readout train. Their dense access to all selected nodes is an
  artificial body interface, not an anatomical motor-neuron mapping.

## Physical control and teacher

Panda from pinned MuJoCo Menagerie, actual position servos and dynamics. Observation
20: q7, qvel7, ee xyz3, target-minus-ee3. Action3: tanh in [-1,1], multiplied by
0.02 m **per axis**, then damped-least-squares IK holding the reset orientation.
Joint commands have step and range limits. 25 physics steps of 2 ms per action =
20 Hz. The gripper stays open. No qpos assignment occurs inside step.

Targets are sampled relative to the initial end effector within ±0.12 m x/y and
±0.10 m z. Initial joints vary by ±0.025 rad. Targets are not feasibility-filtered;
report teacher success alongside learned results. Success means <0.02 m for five
consecutive control steps; horizon 100 = 5 s. This is deliberately a small task.

Teacher a = clip((target-ee)/.02, -1, 1), using the same IK/servo/dynamics interface.
Learning uses sequence behavior cloning, masked padded trajectories and truncated
BPTT. Teacher actions are **not** mixed into learned closed-loop evaluation.

## Comparisons and leakage prevention

- Train 96 episodes (seeds 0..95); validation 16 (10000..10015); test 24
  (30000..30023). Whole episodes are disjoint, distribution is shared. The 20000-series
  targets exposed an IK rotation-frame defect and are now development regression
  targets, not held-out evidence. The initial defective run is retained separately.
- Training observations alone fit normalizer statistics (std floor 0.05).
- 35 epochs, best validation imitation-MSE checkpoint, Adam .002, batch16, TBPTT20,
  gradient-norm clip1. Hidden states start at zero each episode. CPU, two threads.
- Training seeds [0,1,2]. Shared demonstration data and test episodes across models.
- Connectome and MLP have identical trainable parameter counts. GRU hidden size
  minimizes count difference; actual counts are reported, not assumed identical.
- Shuffle graph: 10 accepted directed double-edge swaps per original edge; preserve
  each node's in/out degree, source sign and raw outgoing weighted strength.
  Incoming weighted strength and graph motifs are not preserved. Report edge overlap.
- Normalization uses the same rule for real and shuffled graphs. Equality of
  node count/degree is not equality of spectral structure or compute cost.
- Zero-action and teacher baselines verify task difficulty/control viability.
- Edges-silenced evaluation removes recurrence from a trained connectome policy
  without retraining adapters; it is a dependence probe, not a matched trained baseline.
- Noise condition: independent Gaussian std0.01 m on the six ee/error entries,
  same seeded noise streams across policies. This is sensor-feature noise, not
  external force or consistent reconstructed geometry.
- Save each evaluation episode's outcome/error/action variation/timing. Policy
  inference time excludes physics, rendering and IK; hardware-specific, CPU only.

## Interpretation and next experiments

Three seeds × 24 paired episodes is an initial check, not statistical proof. It does
not establish sample efficiency: that requires a preregistered data-budget sweep,
more independent seeds and uncertainty estimates. The teacher is simple and target
error is provided directly, so an MLP may solve this task without biological priors.
No result is silently replaced because it is negative. The demo always uses first
training seed and first six consecutive test episodes, not selected successful trajectories.

Next: data-budget curves; retrained disconnected/leaky-dense controls; graph-size
and topology ablations; OOD target tests; latency/scaling; only then harder sensing,
grasping, RL and hardware. Hardware needs an independent safety controller and
validation; the simulation command bounds are not a safety certification.
