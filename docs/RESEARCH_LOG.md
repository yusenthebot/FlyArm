# FlyArm research log

A dated record of every experiment, written for the paper.
Each entry states the question, the method with its commit, config and run directory, the result with counts, the reading, and what it changes.
Negative and inconclusive results stay in the log.
Numbers marked exploratory come from scratch probes that were later superseded by committed scripts; the committed version is authoritative.

## Claims ledger (kept current)

| Claim | Status | Evidence |
|---|---|---|
| The complete MaleCNS (166,700 neurons, 10.5 M edges) runs as a frozen real-time controller on a laptop | supported | E1 |
| Control is graph-mediated: removing every edge removes the skill | supported (pick-place, kitchen, dexterous seed 0: 0/64) | E3, E11, E22, dexterous track |
| The skill needs the connectome's own state across control steps | supported for pick-place and dexterous seed 0 (0/64); kitchen depends on the run (E11 yes, standardized readout 23.8 of 25 without state) | E3, E11, E22, dexterous track |
| Measured wiring beats a degree-preserving shuffle | not established: pick-place first protocol over 6 seeds trends ahead (lift 72% vs 56%, seed-level p 0.125) after seeds 3 to 5 failed to replicate seeds 0 to 2; kitchen fly above shuffle in 2 of 2 seeds; dexterous seed 0 tie (62 vs 63 of 64); v2 (E24) and dexterous seeds 1 and 2 running | E3, E11, E21, E24, dexterous track |
| The fly controller solves more than one kitchen task | not yet: one task (microwave); not fixed by longer memory (E20), tracker DAgger (E19) or a wider standardized readout (E22, whose kettle scores are shoves) | E4, E11, E19, E20, E22 |
| RL on the frozen connectome improves a skill | supported for lifting (18/24 to 24/24), not for placing | E12 |
| RL on the frozen connectome generalizes to unseen physics | partly: lifting generalizes to heavier cubes, but equally through a shuffled connectome; heavy-cube placement favours the measured wiring late in training (15.1 vs 6.3 of 24), nominal and far goals favour the shuffle; one PPO run per wiring | E17 |
| The frozen connectome controls a dexterous hand | supported for seed 0: in-hand rotation 62/64 (teacher 64/64); a matched GRU does better (64/64, faster, 331 vs 251 of 512 on transfer) | dexterous track |
| The rate model holds information for seconds | only near critical recurrent gain (0.99); about 0.2 s at the default 0.8 | E15, E16 |

## 2026-09-20 to 21: B1a and B2, first protocol

### E1. Complete connectome as a controller backend (commit 0712ab0)
Question: can the complete MaleCNS run as a frozen recurrent controller in real time on Apple Silicon?
Method: CSR pack of all 166,700 annotated non-glia neurons and 10,520,377 edges with at least 3 contacts (104.4 M synapses, 84% kept), weights are synapse counts times transmitter sign (acetylcholine +1, GABA and glutamate -1, others 0), row-normalized; rate dynamics h <- 0.5 h + 0.5 tanh(I + 0.8 W h), 3 updates per control step; custom Metal sparse kernel with a transposed-CSR backward pass in MLX.
Result: go/no-go passed (runs/whole-brain-check-001.json): bitwise determinism, no input-to-output bypass (with every edge removed, outputs are exactly zero), stable over 400 steps, 1.5 GiB, about 4 ms per control step (250 Hz).
Sign classes: 103,720 excitatory, 51,371 inhibitory and 11,609 zero-sign neurons whose 332,304 outgoing edges carry no signal.

### E2. B1a reach (runs/whole-brain-reach-001)
Result: 72/72 held-out reaches for the connectome, the shuffle and the GRU; edges off 0/72, direct input-to-output synapses only 72/72, state reset every step 66/72.
Reading: reach saturates every controller and is solved by direct synapses, so it cannot discriminate topology.

### E3. B1a pick-and-place, 3 seeds (runs/whole-brain-pick-place-001, -002)
Method: 1,846 ascending neurons in, 1,314 descending and 708 VNC motor neurons out; behavior cloning from a scripted teacher (100% success) then 2 DAgger rounds; 24 held-out episodes per checkpoint.
Result: lift 58/72 connectome, 27/72 shuffle, 33/72 GRU; stable place 14/72, 10/72, 4/72.
Lesions of the connectome checkpoints: edges off 0/72 lifts, state reset every step 0/72 lifts, direct synapses only 37/72 lifts.
Statistics (src/flyarm/whole_brain/stats.py): paired episodes 36 wins and 5 losses against the shuffle (exact binomial p = 3.9e-7); seed level 3 of 3 seeds better, exact sign-flip p = 0.125 (the minimum possible with 3 seeds).

### E4. B2 FrankaKitchen, first protocol (runs/flyleg-kitchen-complete-001, -proprio-001, -mixed-001)
Method: the Franka as the fly's left front leg: 23 left front-leg proprioceptors and 4,868 head sensory neurons in, 68 left front-leg motor neurons out; MSE behavior cloning, single-step actions, lowest-validation-loss checkpoint.
Findings while building it: behavior cloning with velocity inputs copies the previous action (copycat) and scores 0, so all controllers see positions only; motor-neuron activity is about 3e-4, so a frozen per-neuron readout scale (inverse RMS) is calibrated before training.
Result (3 seeds x 40 episodes, D4RL score): fly 0, shuffle 0, MLP 8.3 +- 14.4, GRU 0; proprioception-only fly 7.3 +- 12.1 with seed 2 opening the slide cabinet in 34/40 episodes, removed by every lesion; mixed split all 0.
Reading: the protocol, not the controllers, failed; the benchmark could not discriminate.

## 2026-09-21: fixing the kitchen protocol

### E5. Protocol diagnosis on the cheap controllers (commits d443056, 6b83213)
Question: which part of the protocol prevents learning?
Method: MLP and GRU only, kitchen-complete, 40 clean episodes; exploratory 3-seed screens, then the committed 5-seed sweep (scripts/kitchen_protocol_sweep.py, runs/protocol-sweep).
Exploratory results (mean over 3 seeds): single-step MSE MLP 8.3 and GRU 0; single-step L1 MLP 30.2 and GRU 10.2; 10-step chunks with L1 MLP 26.0 and GRU 25.0; 300 instead of 100 epochs made the MLP worse (seed 0: 65 to 0); kitchen-partial for 5 epochs gave 0, 0, 0, 15, 25.
Reading: L1 matters most, chunking helps the recurrent controller, and longer training overfits the 17 demonstrations.

### E6. The task is learnable: demonstration tracker (commit 6b83213)
Method: nearest demonstrated state (robot and object joint positions) plus a proportional joint correction, built only from the 19 demonstrations (src/flyarm/benchmarks/kitchen_expert.py).
Result (10 episodes each): 100 clean, 100 under 10x observation noise, 100 from 0.1 rad joint offsets; 87.5 clean without the correction.
Reading: the data supports the whole task; parametric imitation is the bottleneck.

### E7. DAgger with the tracker, pure learner rollouts (exploratory, 1 seed)
Result: MLP 67.5 after behavior cloning, then 0, 0, 36 and 26 after four DAgger rounds.
Reading: negative; learner rollouts that fail dwell far from the demonstrations, where nearest-neighbour labels are discontinuous.

### E8. DART with the tracker (exploratory)
Method: 200 tracker episodes with action noise 0.1 (the tracker still completes 3.6 tasks on average), labels are the clean actions.
Result: single-step L1 MLP 53, 7.5, 50 with validation-loss selection; 10-step chunks 0, 46, 0.
Reading: inconclusive; not adopted.

### E9. Checkpoint selection is the hidden failure (commit 6b83213)
Finding: within one run, neighbouring epochs swing between 0 and 2 or more completed tasks, and the lowest validation loss on the two held-out demonstrations does not predict closed-loop success.
Method: keep the checkpoint with the most tasks completed on 5 validation episodes whose environment seeds (50,000+) are disjoint from the test seeds (0 to 39), scored every 5 epochs.
Exploratory result (3 seeds): MLP single-step 65, 25, 40.6; MLP with DART 42.5, 58.75, 50; GRU single-step 25, 25, 17.5; GRU 10-step chunks 38.1, 50, 50.

### E10. ACT reference (commit d443056)
Method: transformer encoder-decoder with a CVAE (7.4 M parameters, z = 0 at test), 10-step chunks, temporal ensemble; a reference for what the data supports, not a fly model.
Exploratory: 2,000 steps gave 55 on seed 0; 10,000 steps with validation-loss selection gave 25, 0 and 2.5.

### E11. Kitchen protocol v2, 5 controllers (runs/flyleg-kitchen-complete-chunk-001, config configs/flyleg-kitchen-complete-chunk.json, running)
Protocol: position features, 10-step chunks with ACT's temporal ensemble, L1, 100 epochs, closed-loop selection every 5 epochs; identical for every controller; 40 test episodes plus OOD (10x robot or object noise, 0.05 and 0.1 rad joint offsets).
Seed 0 result: ACT 100, GRU 57.5, MLP 31.2, fly 25.0 (microwave in every episode), shuffle 0.
Fly lesions, seed 0: edges off 0, direct synapses only 0, deafferented leg 0, head senses removed 0, state reset every step 0.
Fly OOD, seed 0: robot noise 8.75, object noise 25, joint offsets 21.25 and 11.25.
Reading: the protocol now separates controllers; the fly skill is graph-mediated and uses both senses and its own state, but it stops after the first task.
Seed 1: ACT 98.1, GRU 39.4, MLP 21.9, fly 21.25 (microwave), shuffle 15.0; the fly is above the shuffle in both seeds (25 vs 0, 21.25 vs 15) and close to the MLP, and seed 2 is running (fly training at 21:30).

## 2026-09-21: reinforcement learning on the frozen connectome

### E12. PPO fine-tuning of the motor decoder, pick-and-place (runs/ppo-pick-place-001, commit cdcb970)
Method: BatchedPickPlace on mjbatch reproduces the single environment step for step (maximum observation difference below 1e-5 over a full teacher episode; hinge-axis Jacobian equals mj_jacSite to 1e-12); PPO in MLX trains only the linear decoder from the 2,022 output neurons plus a per-action exploration scale; encoder and connectome frozen; privileged critic only during training; 128 environments, 600 iterations (4.9 M steps), start from the B1a connectome seed-0 checkpoint.
Throughput on the shared M5 Max: about 2,000 environment steps per second (connectome about 3,200, physics about 9,000 with 18 threads).
Result on the 24 B1a test seeds (batched evaluation of the start checkpoint: place 4, lift 18, grasp 24): lift reached 24/24 by iteration 60 and stayed at 24/24 from iteration 240 to 600; place fluctuated between 1 and 7 (4 at iteration 600).
Reading: RL through the fixed connectome reliably improves lifting but not placing.
Caveat found: the run's "best" checkpoint (place 7) was selected on the test seeds; checkpoint selection must use validation seeds, and only the final or validation-selected checkpoint may be reported.

### E13. Harder pick-and-place variants: baselines of the imitation checkpoint (commit 1d15377)
Method: TaskVariant (per-episode cube mass and grip friction from a separate random stream, wider goal range, goal blanked after k steps); the B1a connectome seed-0 checkpoint on the 24 test seeds.
Result (place / lift / grasp): nominal 4/18/24; mass 3 to 6x 1/18/24; friction 0.15 to 0.3x 7/22/24; goals within 18 cm 3/15/22; goal visible 10 steps 0/5/12; friction 0.04 to 0.07x 0/0/21; friction 0.08 to 0.12x 0/0/21; mass 6 to 10x 0/13/24.

### E14. Memory timescale of the rate model (exploratory, then E16)
Method: two copies of the pick-place controller receive different goal inputs for 10 control steps and identical inputs afterwards.
Result: their largest state difference falls about 5x per control step: 2.0, 0.62, 0.12, 0.027, 0.009 at 0 to 4 steps after the inputs converge, 1e-5 after 10 steps.
Reading: at the default gain the connectome holds information for about 0.1 to 0.2 s; the goal-memory task was withdrawn because a trained decoder cannot add memory the dynamics lack.

### E15. Pathway gain into the front-leg motor neurons (scripts/connectome_probes.py, docs/results/connectome-probes.json)
Method: random +-0.5 current on every neuron of one input population for 10 control steps; RMS of the 68 left front-leg motor neurons.
Result: leg proprioceptors (23) 1.2e-3; head sensory (4,868) 3.5e-4; all visual projection neurons (9,201) 1.1e-4; LC and LPLC visual projection neurons (4,581) 5.7e-5; ascending neurons (1,846) 1.1e-2; descending neurons (reference) 1.6e-2.
Reading: rerouting the scene through visual projection neurons would not help; ascending neurons are the strongest non-motor route.

### E16. Recurrent-gain sweep (same script and file)
Result (head to motor RMS, proprioceptive to motor RMS, largest residual state 1, 5, 10, 20 control steps after all inputs stop):
gain 0.8: 3.0e-4, 1.3e-3, 0.52, 0.13, 0.027, 0.0011.
gain 0.9: 4.1e-4, 1.5e-3, 0.60, 0.25, 0.11, 0.023.
gain 0.95: 4.7e-4, 1.6e-3, 0.64, 0.33, 0.19, 0.084.
gain 0.99: 5.2e-4, 1.7e-3, 0.67, 0.39, 0.28, 0.19.
Reading: near-critical gain extends the connectome's memory from about 0.2 s to seconds with the wiring unchanged, while pathway gains change by less than a factor of two.

### E17. PPO with physics randomization (runs/ppo-pick-place-randomized-001, configs/ppo-pick-place-randomized.json)
Method: as E12, trained on cube mass 1 to 4x, grip friction 0.15 to 1x, goals within 14 cm; 800 iterations (6.6 M steps); evaluated every 25 iterations on held-out variants with the 24 test seeds.
Result, final checkpoint (iteration 800, no selection), place / lift with the start checkpoint in brackets: nominal 4/24 (4/18); training distribution 10/24 (5/16); held-out mass 6 to 10x 17/23 (0/13); held-out goals within 18 cm 2/22 (3/15); held-out friction 0.08 to 0.12x 0/0 (0/0).
Reading: through the frozen connectome, RL raised lifting to near ceiling everywhere, including cubes heavier than any trained (13 to 23 of 24), and doubled placement on the training distribution.
The large placement gain on heavy cubes is partly physical: a heavy cube does not bounce or slide on release, and the light nominal cube stays at 4/24.
Far goals and low friction did not improve.
This run predates validation-seed selection, so only final-iteration numbers are reported.
Control: the same run on the degree-preserving shuffle (runs/ppo-pick-place-randomized-shuffled-001) started after it, with validation-seed selection.
Interim control, same test seeds at matched iterations 300 to 375 (place of 24, connectome then shuffle): nominal 2-4 vs 4-6, training distribution 3-7 vs 3-8, heavy 4-8 vs 6-9, far goals 2-3 vs 4-7; lifting is 15 to 24 of 24 for both.
Interim reading: at mid-training the shuffled connectome gains from PPO at least as much as the measured one, so the E17 gains are not yet evidence for the topology; the comparison at iteration 800 decides.
Final, shuffle (runs/ppo-pick-place-randomized-shuffled-001, complete): start checkpoint place / lift nominal 3/11, training distribution 7/21, heavy 2/4, far goals 1/13; iteration 800 nominal 5/24, training distribution 5/24, heavy 5/22, far goals 2/22, low friction 0/0.
Late-training means over the 9 evaluations from iteration 600 to 800 (place of 24, connectome vs shuffle; the evaluations reuse the test seeds, so they are not independent): heavy 15.1 vs 6.3 (connectome 11 to 18 in every one, shuffle 4 to 9), training distribution 7.7 vs 3.3, nominal 2.7 vs 4.9, far goals 2.0 vs 4.4.
The shuffle's validation-selected checkpoint (iteration 75, training distribution) places 13/24 on the test seeds, above anything the connectome run reached, and then the shuffle's placement on the training distribution declines.
Reading: PPO through a shuffled connectome also turns lifting to near ceiling (lift 22 to 24 of 24), so that part is not topology-specific; the measured wiring keeps a consistent late advantage on heavy cubes and the training distribution while the shuffle is better on nominal and far goals.
With one PPO run per wiring this is not a topology claim; it needs PPO replicates (several PPO seeds and both shuffles) before the paper states any difference.

### E18. Beta-mixed DAgger with the tracker on the kitchen, MLP probe (runs/flyleg-kitchen-dagger-dev-001, commit 90cadbf, stopped)
Method: from behavior cloning, four DAgger rounds of 20 episodes in which the teacher acts with probability 0.5, 0.25, 0.125, 0.0625, every visited state labelled by the tracker.
Result: seed 0 20.6 (behavior cloning alone 31.2); rollouts completed 3.8, 2.1, 1.4 and 1.75 tasks as the teacher share fell.
The run was stopped after seed 1 once E19 was negative too.

### E19. Beta-mixed DAgger on the fly controller (runs/flyleg-kitchen-dagger-fly-dev-001, commit 3eddaf6)
Method: from the E11 fly seed-0 checkpoint (25, microwave), three rounds of 16 episodes and 10 epochs with teacher shares 0.5, 0.25, 0.125.
Result: 0 on the 20 test episodes and 0 in every OOD variant; rollouts completed 3.0, 0.38 and 0.06 tasks; the best validation score in the three rounds was 0, 1 and 0 tasks.
Reading: negative; further training on tracker-labelled states destroys the fragile microwave skill even in the first, mostly teacher-driven round.
Implementation flaw found: each round kept its own best checkpoint and the last round's was used; the experiment now keeps the best over all phases, including the starting checkpoint (commit after 7f9734a).
Decision: the tracker-teacher DAgger line is closed.

### E20. Near-critical gain on the kitchen (runs/flyleg-kitchen-gain099-dev-001, commit b37c927)
Question: does the connectome's longer memory at gain 0.99 let the fly controller go beyond the first kitchen task?
Method: E11 protocol with recurrent gain 0.99, fly only, seed 0, 20 test episodes.
Result: 25 (microwave in every episode, no other task); best validation score during training 1 task, as at gain 0.8; final training L1 0.106, as at gain 0.8.
Reading: negative; memory is not what stops the fly controller after the first task.

### E21. Pick-and-place topology replication (runs/whole-brain-pick-place-003, complete; -004 cancelled)
Method: seeds 3 to 5 with two independent shuffles each (replicate r uses shuffle seed seed + 17000 + 1000 r), then a second shuffle for seeds 0 to 2.
Partial (lift / stable place of 24; shuffles listed as #1, #2): seed 3 connectome 9/5, shuffles 15/11 and 12/3, GRU 17/3; seed 4 connectome 17/13, shuffles 22/12 and 22/4, GRU 18/1; seed 5 connectome 20/3, shuffles running.
Final seed 5 shuffles: 24/4 and 12/1 (lift / place), GRU 14/0.
Seeds 3 to 5 alone (two shuffles each; scripts/topology_report.py): lift connectome 64% vs shuffles 74%, 1 of 3 seeds better, sign-flip p 0.88, episodes 19 measured-only vs 34 shuffled-only; place 29% vs 24%, 2 of 3 seeds better, p 0.38.
First protocol pooled, seeds 0 to 5 (docs/results/pick-place-first-protocol-topology.json): lift 72% vs 56%, 4 of 6 seeds better, seed-level p 0.125, episode-level p 0.061; place 24% vs 19%, 5 of 6 better, p 0.14 and 0.096; grasp 86% vs 89%, no difference.
Reading: the strong advantage of seeds 0 to 2 (E3) did not replicate; pooled over six seeds the measured wiring trends ahead on lifting and placing but no test reaches 0.05.
Method lesson: the E3 episode-level p of 3.9e-7 treated 72 episodes as independent, but between-seed and between-shuffle variance is large (seed 5 shuffles lift 24 and 12 of 24), so the seed-level test is the primary one and the paper reports both.
This protocol's teacher mislabels some lifted states (E23); the clean and final comparison is E24.

## Parallel tracks (agents on their own branches)

- Multi-task (branch feat/multitask-manipulation): four skills (pick and place, side push, lift and hold, stack on a platform) with recovering scripted teachers and a preregistered protocol with interpolation, extrapolation, compositional and unseen-object splits (docs/MULTITASK.md on that branch).
- Long-horizon (branch feat/long-horizon): multi-object sequences whose progress is visible in the scene, because the default rate model's memory is short.
- Dexterous hand (branch feat/dexterous-hand): the LEAP hand with four fly legs as fingers; privileged PPO teacher 64/64 in-hand rotation episodes without drops; a GRU distilled from it 60/64; the fly runs wait for GPU.

## Engineering that the paper relies on

- Batched MuJoCo (mjbatch 0.1.1, MuJoCo 3.13.0) with a step-for-step equivalence test against the single environment (tests/test_batched_pick_place.py).
- Local dashboard of every run, curve, log and rollout (flyarm dashboard, port 8780).
- Control-network figure (docs/figures/control_network-figure.png), QA-gated.

### E22. Readout ablation on the kitchen
Question: is the fly controller underfitting because it reads only 68 motor neurons?
Evidence for the question: after behavior cloning the fly's training L1 is 0.106, against 0.068 for the GRU, 0.075 for the MLP and 0.043 for ACT (E11), and neither longer memory (E20) nor corrective data (E19) helped.
First attempt (runs/flyleg-kitchen-descending-dev-001, stopped): reading the 68 motor neurons plus all 1,314 descending neurons with scale-only calibration, training L1 stayed near 0.9 from epoch 1: the 1,382 unit-RMS outputs share a strong common mode and saturate the tanh decoder.
Fix (commit after 5a9f2d9): an optional standardized calibration that subtracts each output's mean activity before dividing by its standard deviation, still a frozen per-neuron affine map.
Design: two runs that differ only in the readout, both standardized, fly only, seed 0, 20 test episodes: runs/flyleg-kitchen-leg-std-dev-001 (68 motor neurons) and runs/flyleg-kitchen-descending-std-dev-001 (68 motor plus 1,314 descending neurons).
Result, 20 test episodes each (both complete): 68 motor neurons (113,536 trained parameters) best validation L1 0.109, microwave in 20/20 and nothing else, score 25, OOD 21 to 25; lesions edges off, direct only, deafferented leg and head sensory deprived all 0, state reset every step 23.8.
68 motor plus 1,314 descending neurons (231,796 parameters): best validation L1 0.817, kettle in 20/20 and nothing else, score 25, OOD 22.5 to 25; every lesion 0.
The kettle completions are not a learned skill: in the recorded episodes (runs/flyleg-kitchen-descending-std-dev-001/videos) the arm flails, its links sweep the kettle from the front-left to the back-left burner without the gripper on the handle, and then the arm swings upward.
Benchmark caveat: the kettle counts as done when its 7-dimensional position and orientation are within 0.3 of the goal, and it starts 0.41 away, so a 10 to 15 cm shove toward the back suffices; kettle scores must be checked on video or with a lift criterion.
Reading: a wider readout is not the fix; with descending neurons the decoder does not fit at all (L1 0.82 against 0.11), and the 68-neuron readout reproduces E11 (microwave only, L1 0.106 there, 0.109 here), so standardizing the readout changes nothing either.
Open: why 1,382 standardized outputs do not train; to probe (per-neuron scale floor, spread of the standardized values, decoder pre-activation) before any further wide-readout run.

### E25. Why the kitchen fly controller stops after one task: the motor interface is low-dimensional (scripts/pathway_depth.py, scripts/linear_response.py)
Question: does scene information reach the 68 left front-leg motor neurons, and how many independent signals can they carry?
Structural depth (docs/results/pathway-depth.json, signed edges only): 60 of the 68 motor neurons are 2 synapses from the head sensory neurons (7 at 3, 1 at 4); from the leg proprioceptors 21 are 1 synapse away and 46 are 2.
Neural steps per control step: steady-state motor RMS under random +-0.5 drive is 3.45e-4 from the head and 1.21e-3 from the proprioceptors whether a control step has 3, 6, 12 or 24 neural updates; more updates only reach it sooner (after 2 control steps at 3).
Linear response (docs/results/linear-response.json): at the rate model's operating point the interface is close to linear (a linear prediction of the motor readout under training-scale drive errs by 9%), so it acts as a fixed matrix M from the 4,891 input neurons to the readout.
For the 68 motor neurons M carries few independent signals: participation ratio 3.8, 5 singular values hold 90% of the energy and 10 hold 99%; the head-sensory block alone has participation ratio 1.6 (3 values for 90%) and 30 times less energy than the proprioceptive block.
Adding the 1,314 descending neurons raises this to 78 values for 90% and 207 for 99%, with a head-sensory block 130 times stronger, but a condition number of 4e7.
The kitchen shuffle (seed 17000) passes more head signal to the leg motor neurons (35 times the energy, 8 values for 90%) and less proprioception, as expected when 4,868 head inputs are rewired at random.
Reading: through its own motor neurons the measured connectome gives the leg a controller with about five usable output directions, dominated by proprioception, and little independent scene information; a multi-stage task needs more, which is the mechanistic account of the one-task ceiling in E11, E19, E20 and E22.

### E26. The wide-readout failure was an optimizer effect, and its fix (scripts/readout_conditioning.py, scripts/readout_warmup_probe.py, commit b1313b8)
Hypothesis falsified first: with shuffled i.i.d. mini-batches of 256 demonstration steps, a decoder-only Adam fit on the untrained seed-0 features trains normally for both readouts and both calibrations (L1 about 0.10 after 3,000 steps, docs/results/readout-conditioning.json).
Reproduced exactly (docs/results/readout-warmup.json): replaying the run's own warm-up (4 episodes, 8-step BPTT windows, clip 1.0, Adam 1e-3) the 1,382-neuron standardized readout's decoder pre-activation spread goes 2.9, 5.5, 7.6, 9.5, 10.6 over the first five updates and the loss jumps from 0.44 to 0.86 after one update, while the 68-neuron readout stays below 1.0.
Mechanism: early Adam steps move every decoder weight by about the learning rate, and within one window of consecutive steps the standardized outputs are strongly correlated, so an update shifts the pre-activation by about lr times the number of outputs, about 1 per update at 1,382 outputs, which saturates the tanh; the gradients then vanish and training never recovers (E22, L1 0.82).
Fix: calibration "unit_norm", the same frozen per-neuron standardization divided by sqrt(outputs); the model class is unchanged, and in the replay the pre-activation spread stays below 0.3 (PCA whitening to 57 components also works, below 1.3).
Runs (fly only, seed 0, 20 test episodes, otherwise as E22): runs/flyleg-kitchen-leg-unitnorm-dev-001 and runs/flyleg-kitchen-descending-unitnorm-dev-001, started 2026-09-21 21:57; the wide readout now trains (epoch 2 L1 0.21, against 0.95 with standardize).
Question they answer: with the optimizer fixed, does reading the brain's 1,314 descending command neurons let the fly controller do more than one kitchen task?

### E23. Stateful-teacher labels in B1a DAgger (found by the multi-task agent, quantified here)
Finding: the scripted pick-and-place teacher keeps its own stage machine, so while labelling learner-driven DAgger states it can still be in "approach" or "descend" after the learner has already grasped and lifted the cube, and it then labels those states "open the gripper".
Quantification over every saved B1a DAgger set (runs/whole-brain-pick-place-001 to -003, 36 files): 337,014 labelled states, 6,353 with both fingers on a cube at least 6 cm above rest, 931 of those (15%) labelled with an open gripper, all in teacher stages 0 or 1.
Reading: 0.28% of all labels, concentrated in the lift and transport phase; every controller in a run received the same labels, so the comparisons stay matched, but absolute B1a scores are probably depressed.
Action: the multi-task branch derives the teacher's stage from the physical state (0 of 1,067 such labels remain, its own demonstrations unchanged); B1a pick-and-place should be rerun with a stateless teacher before its numbers go into the paper, with the current runs kept as the first protocol.

## Parallel tracks, status 2026-09-21 evening

### Multi-task (branch feat/multitask-manipulation, commits 92dba87 to f790242)
Teachers (50 episodes per cell): pick and place 50/50 everywhere; push 50/50 train, 49/50 iid, worst held-out cell 46/50; lift and hold 50/50, worst 48/50; stack 50/50, worst 46/50.
Scene changes needed for pushing: elliptic friction cones and floor friction 0.6.
Early controls, seed 0, 12 episodes per cell (successes): GRU iid 5/48, interp 12/48, extrap 3/48, compositional 3/24, object 6/48; MLP 27/48, 26/48, 10/48, 0/24, 27/48; neither combines a skill with a region it never trained in.
The connectome run (runs/multitask-001, about 15 to 20 h) waits for the GPU.

### Long-horizon (branch feat/long-horizon)
Tasks: tower, sort and clear with four colour-coded cubes, 600 to 1,000 control steps; teacher 900/900.
Design question (decided below): the observation includes a memoryless sub-task cue (the cube to handle now, recomputed from the scene every step), because without it no controller learned to pick the next cube in a pilot (MLP correct first pick 0/36, with the cue 27/36); a cue-silenced lesion keeps the brain's own sequencing testable.
Early controls, seed 0, 36 test episodes per split (progress): GRU iid 0.25 (4/36 full), MLP 0.11 (1/36); both near 0 without the cue; tower about 0 for both.
The connectome and shuffle runs (12 to 24 h) wait for the GPU.
User decision (2026-09-21): keep the cue for the main run and add a no-cue control (config cue=false zeroes the 17 cue entries, same architecture, parameters, teacher and labels; commit 328222d on feat/long-horizon).
Reading rule 8 on that branch: a controller sequences on its own if its no-cue iid progress is at least half its cued progress and its first pick is right in at least half of the episodes.
Cued controls, 20:00 (iid progress, full successes of 36, progress with the cue silenced): GRU seed 0 0.25, 4, 0.05; GRU seed 1 0.05, 0, 0.03; MLP seed 0 0.11, 1, 0.00; MLP seeds 1 and 2 training; the seed-to-seed spread of the GRU is large.
The connectome queue (runs/long-horizon-queue-brain.log) has waited at its GPU gate since 18:56.
Cued controls complete (runs/long-horizon-001-controls, seeds 0 to 2; tables in docs/results/long-horizon-001-controls.json and .md, commit 3ea7e5f on feat/long-horizon): iid progress per seed GRU 0.25, 0.05, 0.12 (mean 0.14, full successes 4/108), MLP 0.11, 0.17, 0.16 (mean 0.15, 6/108).
Silencing the cue drops both to about 0 (GRU 0.03, MLP 0.00); silencing ranks, placed flags or the task code changes little; unseen target placements are the weakest split (0.06 to 0.07); the no-cue controls started 21:14.

### Dexterous hand (branch feat/dexterous-hand)
LEAP hand, four fly legs as fingers; privileged PPO teacher frozen at iteration 925 (15.2 M steps): 64/64 rotation episodes, 0 drops, 50 rad in 20 s on selection seeds; held-out transfer of the teacher around iteration 500 (selection seeds): small cube 54/64, sphere 53/64, cylinder 49/64; distilled GRU pilot 60/64.
The main fly run (connectome, shuffle, GRU; seeds 0 to 2) started 2026-09-21 19:30 after its GPU gate was raised from fewer than 3 to fewer than 7 other full-connectome runs.
Connectome seed 0 (runs/hand-dexterous-001/connectome-0, 86,046 trained parameters, 64 test episodes): 62/64 successes, 2 drops, mean rotation 38.0 rad, against the teacher's 64/64, 0 drops and 49.7 rad on the same episodes.
Lesions (successes of 64): edges off 0 (no rotation); state reset every control step 0 (no rotation), so the controller uses the rate model's short memory; front-right leg (thumb) deafferented 0; middle-right leg (ring) 36, rotation 6.8 rad; front-left (index) 63; middle-left (middle) 62; head sensory silenced 47.
Zero-shot object transfer, connectome vs teacher: small cube 0 vs 28, large cube 29 vs 38, light 51 vs 63, heavy 64 vs 64, slippery 16 vs 34, grippy 56 vs 63, cylinder 16 vs 38, sphere 19 vs 32.
Reading: the frozen connectome with linear maps learns in-hand rotation close to the teacher, and the recurrent connectome is necessary for it (edges off gives 0); whether the measured wiring matters is open until the seed-0 shuffle and GRU finish (then seeds 1 and 2, expected 2026-09-22 01:00 to 02:00).
Video: runs/hand-dexterous-001/connectome-0/rollout.mp4, episode 200000 turns 29.5 rad (1,687 degrees) in 20 s; featured on the dashboard.
Shuffle seed 0 (degree-preserving, same parameters and data): 63/64 successes, 1 drop, 35.2 rad, so on the training object the measured wiring gives no advantage.
Transfer, connectome vs shuffle: small cube 0 vs 0, large 29 vs 8, light 51 vs 51, heavy 64 vs 59, slippery 16 vs 10, grippy 56 vs 54, cylinder 16 vs 16, sphere 19 vs 18 (251 vs 216 of 512 in total, most of the gap from the large cube).
Reading: one seed; the only visible difference is transfer to a larger cube, to be tested on seeds 1 and 2 with the episode-level binomial and the seed-level sign-flip test before any claim.
Parameter-matched GRU seed 0 (86,531 parameters, same data): 64/64, 0 drops, 50.4 rad, at the teacher's level; transfer 331/512 (small cube 10, large 27, light 63, heavy 64, slippery 37, grippy 63, cylinder 44, sphere 23).
Reading: on seed 0 the GRU beats both connectome controllers, most clearly in rotation speed (50.4 against 38.0 and 35.2 rad) and in transfer to slippery, cylindrical and small objects; the dexterous result is that the frozen connectome can do in-hand rotation, not that it does it better than a trained recurrent network.

### E24. B1a pick-and-place protocol v2 (configs/whole-brain-pick-place-v2a.json and -v2b.json, running)
Change from the first protocol: the stage-resynchronized teacher (teacher "resync", commit 5a9f2d9), whose demonstrations are bit-identical to the first protocol's on six compared episodes and which keeps squeezing a cube a learner has already lifted.
Design: seeds 0 to 5, each training the connectome, two independent degree-preserving shuffles and a parameter-matched GRU on identical data; lesions as before; statistics per E3 (seed-level sign-flip and episode-level binomial).
Run v2a (seeds 0 to 2) started 2026-09-21 21:35 when run 003 finished, v2b (seeds 3 to 5) follows (scripts/pick_place_v2.sh); run 004 (a second shuffle for the first protocol's seeds 0 to 2) was cancelled because v2 supersedes it.

## Housekeeping 2026-09-21 evening
Superseded rollout media (67 files: the first kitchen protocol, most proprioception-only clips, page copies, the 256-node prototypes and a duplicate) moved to runs/_archive with a manifest by scripts/archive_media.py; at the user's request the archive (148 MB, manifest included) then went to the macOS Trash, so these files are no longer part of the record.
The obsolete live views on ports 8769, 8770 and 8771 were stopped; only the dashboard (8780) stays.
Stale waiters were stopped: one on the stopped E18 run and five duplicate waiters of the long-horizon agent.
The control-only protocol sweep (runs/protocol-sweep, 46 of 50 settings done, the 4 left are GRU with 10-step chunks, seeds 1 to 4) is paused with SIGSTOP to give the fly runs the GPU; resume with kill -CONT on its Python process, or rerun the script, which skips finished settings.
The dashboard's rollout page now shows curated featured rollouts (docs/featured-videos.json) and groups every other video by area with readable titles; smoke tests and drafts are hidden by default.
PPO before-and-after videos (flyarm rl record) render the imitation checkpoint and the final PPO checkpoint on the same seeds and task variant.
