# Why complex manipulation fails: an architecture-level analysis (2026-09-25)

The kitchen (four tasks, 100 in the official environment) and pick-and-place (95% placement) work with the frozen connectome; the articulated multi-step manipulation benchmark does not.
This note separates the causes by layer, from the evidence in docs/RESEARCH_LOG.md (E55, E56 and the entries after them) and docs/MANIPULATION_ENV.md.

## Layer 1: the connectome adds a fixed, shallow projection and nothing more

With the whole-body B1a interface the readout is dominated by one synaptic layer from the 1,846 ascending neurons; currents arriving by longer paths are about 1e-3 (median positive input 0.0003).
With linear maps in and out, the connectome therefore behaves as a linear controller: frame-wise fits of 0.195 to 0.216 against 0.211 for a linear policy on the raw observation, and an MLP at 0.021.
Neither a rectified firing-rate transfer, more neural steps per control step (3, 6, 10), a recurrent gain of 0.99 nor ten times the input drive changes this; a degree-preserving shuffle and the direct synapses alone match the measured wiring.
With a nonlinear sensory encoder in front of it the capacity reaches the MLP level (0.166 held out against 0.145), and the shuffle again matches (0.159).
Cause: synaptic weights normalized per neuron over a graph of 166,700 neurons, driven through 1,846 input neurons, attenuate geometrically with path length, so only the first synapse carries usable signal.
Consequence: the connectome cannot supply the nonlinear computation the task needs, and whatever the controller achieves, a shuffled graph achieves too.

## Layer 2: the task demands discrete decisions that a smooth regressor cannot make

The scripted teacher is a phase machine (approach, descend, close, move, lift, carry, lower, release, retreat), and its commands switch at precision gates.
Measured on 42,575 teacher steps (scripts/manipulation_phase_linearity.py, docs/results/manipulation-phase-linearity.json), a single linear map fits its commands to L1 0.258 (0.595 on held-out episodes), one linear map per skill to 0.214 (0.574), one linear map per teacher phase to 0.094 (0.316), and one per skill and phase to 0.068 (0.245).
So about two thirds of the error of a linear controller is the switching between phases; within a phase the command is close to a linear function of the observation.
This is why every learner, the MLP and GRU included, stalls near the gates (stationary in 30 to 54% of steps, a few millimetres outside a gate) and why each fix so far has been a change to a gate.

## Layer 3: the control loop is flat and fine-grained

One flat policy chooses every 5-D Cartesian step for 1,100 to 2,600 steps, so a small bias compounds, and every discrete decision is made implicitly, step by step, by a function approximator.
The kitchen worked where this does not because its horizon is 280 steps of smooth joint-velocity motion with no precision gates, and pick-and-place because it is one object and 400 steps.

## Layer 4: the supervisor is hand-written and discontinuous

The teacher's gates and floors were designed for its own trajectories; the learner visits states the teacher never visits (a hand already at a handle, 3 mm outside a gate), where its labels were contradictory or near zero until the recent fixes.

## What follows for the architecture

1. Make the discrete structure explicit instead of asking the controller to discover it: extend the memoryless cue from the current skill to the current motor phase, computed from observable scene predicates (aligned, at height, grasped, above target), so the controller only has to produce the within-phase command, which is close to linear. This matches the insect architecture (descending command neurons select motor programs) and keeps the connectome in the loop; it needs a no-phase-cue control, as the skill cue has.
2. Replace the hand-written phase machine as the supervisor with smooth privileged experts (per-skill policies trained by reinforcement learning on privileged state), whose commands are smooth functions of state and so imitable.
3. Shorten the effective horizon with temporal abstraction (action chunks or waypoint actions), which reduces compounding error.
4. Separately, and only if the paper is to claim that the wiring matters: study signal propagation in the rate model (the weight normalization, the input coverage, the time constants), because under the present normalization the measured wiring cannot express more than one synapse.
