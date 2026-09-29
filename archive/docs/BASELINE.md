# MaleCNS baseline: what is real and what is abstract

FlyArm's baseline is not a network drawn to look like a fly brain. It is a deterministic
subgraph of measured neurons and synaptic contacts from **MaleCNS v1.0**, the complete
adult male *Drosophila* central nervous system connectome released by HHMI Janelia,
Cambridge/MRC LMB and Google Research. The upstream resource contains more than 166,000
neurons and 125 million synaptic connections and is licensed CC BY 4.0.

Official sources:

- [MaleCNS project and interactive data](https://male-cns.janelia.org/)
- [MaleCNS bulk downloads](https://male-cns.janelia.org/download/)
- [Google Research release note](https://research.google/blog/a-connectomics-milestone-mapping-the-complete-male-fruit-fly-brain/)
- [Janelia project description](https://www.janelia.org/project-team/flyem/male-cns-connectome)

## Immutable source chain

`flyarm fetch` downloads three official v1.0 flat-connectome objects by their immutable
Google Cloud Storage generation, verifies their byte size and transport MD5, then checks
an independently pinned SHA-256. The derived graph embeds that manifest. A run refuses
to start unless the graph passes all of the following:

1. dataset and recipe versions match;
2. all three source SHA-256 values match the code pins;
3. the ordered numeric graph content hashes to
   `7a5018c5481d14307e1efec305f4f448648545e5feda7caf9297dab518f4da5d`;
4. it contains exactly 256 measured neuron Body IDs and 4,678 measured directed edges.

The live UI reads the official `somaLocation`, `type` and `superclass` fields for those
same Body IDs. It displays a measured soma-coordinate layout, not an invented anatomical
mesh and not a force layout. Clicking a node exposes the source Body ID and annotation.

## Preregistered subgraph and robot interface

The 256 nodes were selected without robot outcomes: four annotated descending cell types
seed deterministic breadth-wise growth by strongest adjacent contact sum. The resulting
class counts are:

| MaleCNS superclass | Nodes | FlyArm role |
|---|---:|---|
| `ascending_neuron` | 23 | only observation-write locations |
| `descending_neuron` | 43 | only action-read locations |
| `cb_intrinsic` | 165 | internal recurrence |
| `vnc_intrinsic` | 24 | internal recurrence |
| `vnc_motor` | 1 | internal recurrence |

The 23 input and 43 output Body IDs are disjoint and stored in `interface.json`; their
ordered interface fingerprint is
`3f2e38c26bfa68af6018eb108c90c2114ad86b4c5d434bc311a7756aac0655d0`.
On the measured graph, 42 of 43 output neurons are reachable from the declared inputs in
one or two directed hops. There is no encoder-to-readout bypass.

These roles are an explicit **engineering proxy**. We do not claim that a fruit fly's
ascending neurons naturally encode Panda joint state or that its descending neurons
naturally command a robot arm.

## Dynamics and evidence levels

The recurrent edge weights come from measured contact counts plus a documented
neurotransmitter-sign approximation. State evolution is a leaky `tanh` recurrence. It is
not a conductance model, spiking simulation, whole-brain emulation or behavioral model of
the fly.

Claims are separated in the UI and results:

| Observation | What it can support |
|---|---|
| trained MaleCNS succeeds; the same trained adapters fail with all graph edges silenced | graph-mediated information flow was necessary for that policy |
| retrained MaleCNS exceeds retrained directed-degree-preserving shuffles | evidence that measured topology helped under this protocol |
| teacher succeeds | the physical task and action API are feasible; says nothing about the learned brain policy |
| a 3D animation moves | visualization only; no scientific claim |

All learned evaluations are closed loop and never corrected by the teacher. Pick-and-place
success requires recorded dual-finger contact, a lift of at least 6 cm, transport to within
3 cm of the goal, physical release, no finger contact and ten stable control steps.
