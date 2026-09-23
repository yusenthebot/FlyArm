# Milestone: the frozen fly connectome solves FrankaKitchen (2026-09-23)

The complete MaleCNS (166,700 neurons, 10.5 M measured connections) stays frozen.
Only a linear map into its ascending sensory neurons and a linear map out of its descending and motor neurons are trained.
With that controller the Franka completes all four tasks of D4RL kitchen-complete (microwave, kettle, light switch, slide cabinet) in every test episode of the official Gymnasium-Robotics environment, including starts it never trained on.

## Headline numbers

Official FrankaKitchen environment, 50 test episodes per condition, every episode run to the 280-step horizon (docs/results/kitchen-milestone-official.json, scripts/kitchen_final_eval.py).
The benchmark score counts tasks ever completed at the benchmark's threshold of 0.3; the strict score counts only elements that end the episode within 0.1 of their goal.

| Controller | Start | Benchmark score | All four tasks | Strict score | Saturated commands |
|---|---|---:|---:|---:|---:|
| Four-task controller (E49) | benchmark start | 100.0 | 50/50 | 50.0 | 47% |
| Four-task controller (E49) | 0.2 rad perturbed | 100.0 | 50/50 | 50.0 | 47% |
| Four-task controller (E49) | 0.3 rad perturbed | 100.0 | 50/50 | 50.0 | 47% |
| Motion-quality controller (E53) | benchmark start | 100.0 | 50/50 | 69.5 | 19% |
| Motion-quality controller (E53) | 0.2 rad perturbed | 97.5 | 48/50 | 70.5 | 11% |
| Motion-quality controller (E53) | 0.3 rad perturbed | 96.5 | 46/50 | 68.0 | 10% |

Training used the benchmark start only; the perturbed starts are generalization.
Blind replay of a demonstration scores 16 to 19 from these perturbed starts (research log E30), so the scores measure closed-loop control, not a memorized action sequence.
For reference, every earlier reward-only run stopped at one task (25), and the published behaviour-cloning result on kitchen-complete is about 65.

Final distance to goal, median over the 50 episodes from the benchmark start (demonstrations end at 0.065, 0.087, 0.008 and 0.021):

| Controller | Microwave | Kettle | Light switch | Slide cabinet |
|---|---:|---:|---:|---:|
| Four-task (E49) | 0.09 | 0.23 | 0.00 | 0.20 |
| Motion quality (E53) | 0.00 | 0.23 | 0.00 | 0.07 |

## Recipe

All stages train only the two linear maps; the connectome is never changed.

1. Imitation (DAgger with a demonstration tracker) on the whole-body interface: 1,846 ascending neurons in, 2,022 descending and motor neurons out; the imitation controller completes the microwave only (25).
2. PPO with a demonstration term (DAPG, Rajeswaran et al. 2018): the squared error to the base run's own demonstrations is added to the PPO loss, so the policy stays near demonstrated behaviour (E46).
3. The completion bonus is paid only in the demonstrations' order, so PPO stops trading the microwave for the kettle; this alone gives microwave then kettle in 20 of 20 perturbed episodes (E47).
4. Potential-based task shaping at weight 10, which pays for progress toward the next element and charges for moving away, unlocks the light switch and the slide cabinet (E48, E49).
5. Motion quality (E50 to E53): a potential for the last stretch from the threshold to the exact goal, penalties for stray contacts (MuJoCo contact sensors on the benchmark model, physics bit for bit unchanged), for moving objects outside the task, for large and abrupt commands, half of each completion bonus held back until the element is within 0.1, and demonstration-state resets (states only, never actions).

About 19 M environment steps for the four-task controller of seed 0 after imitation.

## Replication

The four-task stage replicates on three PPO seeds from the same imitation checkpoint (batched environment, 20 test episodes from 0.2 rad perturbed starts): seed 0 100.0 with all four in 20 of 20 (50 of 50 in the official environment), seed 1 100.0 with 20 of 20, seed 2 97.5 with 18 of 20.

## Videos

- runs/ppo-kitchen-dapg-ordered-potential-continue-001/videos/four-tasks-perturbed-0.3rad-grid.mp4: four episodes from 0.3 rad starts, all four tasks each.
- runs/ppo-kitchen-quality-demoreset-001/videos/before-after-quality.mp4: the same perturbed episode before and after motion-quality training.

## Open items

- The kettle is struck from the side and left at about 0.23 from its goal instead of being lifted onto the back burner as in the demonstrations; it passes the benchmark's 0.3 but not the strict 0.1. From demonstration mid-lift states the policies reach 0.1 about half the time but do not hold it (E53, E54).
- Stray contact falls from 0.14 to 0.09 to 0.11 of control steps but is not zero, and the motion-quality controller loses a few episodes from 0.3 rad starts (46 of 50 all four).
- Controls for the reinforcement-learning stage are not yet run: a degree-preserving shuffle of the connectome and a parameter-matched GRU or MLP through the same recipe; until then the claim is that the frozen connectome can be trained to solve the task, not that its wiring is what makes it solvable.
- Checkpoint selection: the trainer selected on the clean start, whose 20 seeds are one deterministic episode; the reported checkpoints were chosen on disjoint perturbed validation seeds, and the paper must state that rule.

Full history: docs/RESEARCH_LOG.md, entries E36 to E54.
