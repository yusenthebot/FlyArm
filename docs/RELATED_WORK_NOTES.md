# Related work notes (survey 2026-09-22, citations still to be verified against the sources)

Every entry below came from a web survey and must be checked against the paper itself before it is cited.

## Connectomes as controllers
- Lappalainen et al. 2024, Nature: connectome-constrained network of the fly visual system (about 45,000 neurons); connectivity and signs fixed, gains and biases trained; no control task.
- Wang-Chen et al. 2024, Nature Methods (NeuroMechFly v2, FlyGym): biomechanical fly body with a connectome-constrained visual network; odor navigation and fly following.
- Jin, Zhu, Zhang and Sui, arXiv:2602.17997: the adult connectome as a message-passing graph trained end to end with deep RL for whole-body locomotion; the connectome's weights are trained, not frozen.
- Eon Systems 2026 (industry report, not peer reviewed): the FlyWire connectome frozen and driving a MuJoCo body without task training.
- Bhattasali, Zador and Engel 2022, NeurIPS (NCAP), and Rodriguez et al. 2024, arXiv:2410.07174: C. elegans-inspired circuit priors trained with RL and evolution strategies; control is a size-matched MLP; sign constraints, not topology, carry the gain.
- Costi et al. 2025, Biomimetics: the larval Drosophila connectome as a frozen echo-state reservoir with a ridge readout; controls are shuffled and random reservoirs at matched spectral radius.
- arXiv:2606.17745: a frozen rate operator from the complete larval connectome; degree and weight statistics set gross dynamics, exact wiring sets input routing.

## Reservoirs trained with reward
- FORCE learning (Sussillo and Abbott 2009, Neuron): recursive least squares on the readout with feedback into the reservoir.
- Evolution strategies and CMA-ES suit a small linear readout and avoid truncated gradients through a chaotic recurrence.

## Benchmarks and budgets
- Relay Policy Learning (Gupta et al. 2019, CoRL): flat RL fails on the four-task FrankaKitchen; demonstrations plus hierarchy plus RL fine-tuning are needed.
- D4RL (Fu et al. 2020): Kitchen was built to be hard for standard RL.

## Controls reviewers expect
- Degree-preserving rewiring with at least five independent shuffles, reported as mean and spread; spectral-radius-matched random networks; cell-class-preserving nulls; parameter-matched MLP and GRU; lesions.
- arXiv:2604.04033 (topological sensitivity): connectome advantages often vanish under fair initialization and degree-preserving controls; compare at matched optimization steps with identical seeds.
