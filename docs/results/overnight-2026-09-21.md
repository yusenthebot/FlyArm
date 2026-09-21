# FlyArm quantitative report

Generated from run artifacts; every number below is read from a results.json file.

## B2: the Franka arm as the fly's left front leg (D4RL FrankaKitchen)

### FrankaKitchen `complete`, fly wired through leg proprioceptors + head senses (`flyleg-kitchen-complete-001`, status: complete)

D4RL normalized score (25 per completed target task, 0-100); mean ± sd over training seeds; published BC reference 65.0 (original D4RL v0).

| Controller | Params | Seeds | Clean | Per seed | microwave | kettle | light switch | slide cabinet | robot_noise_x10 | object_noise_x10 | joint_offset_0.05rad | joint_offset_0.1rad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fly CNS as left front leg (MaleCNS) | 107,947 | 3 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 | 0% | 0% | 0% | 0% | 0.4 ± 0.7 | 0.0 ± 0.0 | 0.6 ± 0.6 | 1.9 ± 1.7 |
| Shuffled CNS, same leg interface | 107,947 | 3 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 | 0% | 0% | 0% | 0% | 0.0 ± 0.0 | 1.0 ± 1.8 | 0.0 ± 0.0 | 0.0 ± 0.0 |
| MLP BC (D4RL architecture) | 76,041 | 3 | 8.3 ± 14.4 | 0.0, 0.0, 25.0 | 33% | 0% | 0% | 0% | 4.4 ± 7.6 | 8.3 ± 14.4 | 6.5 ± 10.6 | 6.7 ± 9.4 |
| GRU, parameter-matched | 107,615 | 3 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 | 0% | 0% | 0% | 0% | 2.9 ± 5.1 | 0.0 ± 0.0 | 2.9 ± 5.1 | 4.6 ± 7.4 |

Lesions of the trained fly controller (same checkpoints, clean evaluation):

| Intact | edges_off | direct_only | deafferented_leg | head_sensory_deprived | state_reset_every_step |
|---|---|---|---|---|---|
| 0.0 ± 0.0 | 8.3 ± 14.4 | 0.0 ± 0.0 | 0.0 ± 0.0 | 8.8 ± 14.1 | 0.0 ± 0.0 |

Zero action: 0.0.

### FrankaKitchen `complete`, fly wired through leg proprioceptors only (`flyleg-kitchen-complete-proprio-001`, status: complete)

D4RL normalized score (25 per completed target task, 0-100); mean ± sd over training seeds; published BC reference 65.0 (original D4RL v0).

| Controller | Params | Seeds | Clean | Per seed | microwave | kettle | light switch | slide cabinet | robot_noise_x10 | object_noise_x10 | joint_offset_0.05rad | joint_offset_0.1rad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fly CNS as left front leg (MaleCNS) | 851 | 3 | 7.3 ± 12.1 | 0.0, 0.6, 21.2 | 0% | 0% | 1% | 28% | 3.3 ± 2.2 | 7.3 ± 12.1 | 2.5 ± 2.9 | 2.9 ± 2.8 |
| Shuffled CNS, same leg interface | 851 | 3 | 0.0 ± 0.0 | 0.0, 0.0, 0.0 | 0% | 0% | 0% | 0% | 1.9 ± 3.2 | 0.0 ± 0.0 | 0.6 ± 1.1 | 0.8 ± 1.0 |

Lesions of the trained fly controller (same checkpoints, clean evaluation):

| Intact | edges_off | direct_only | deafferented_leg | state_reset_every_step |
|---|---|---|---|---|
| 7.3 ± 12.1 | 8.3 ± 14.4 | 0.0 ± 0.0 | 8.3 ± 14.4 | 0.0 ± 0.0 |

Zero action: 0.0.

### FrankaKitchen `mixed`, fly wired through leg proprioceptors + head senses (`flyleg-kitchen-mixed-001`, status: complete)

D4RL normalized score (25 per completed target task, 0-100); mean ± sd over training seeds; published BC reference 51.5 (original D4RL v0).

| Controller | Params | Seeds | Clean | Per seed | microwave | kettle | bottom burner | light switch | robot_noise_x10 | object_noise_x10 | joint_offset_0.05rad | joint_offset_0.1rad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Fly CNS as left front leg (MaleCNS) | 107,947 | 1 | 0.0 | 0.0 | 0% | 0% | 0% | 0% | 0.0 | 0.0 | 0.0 | 0.0 |
| Shuffled CNS, same leg interface | 107,947 | 1 | 0.0 | 0.0 | 0% | 0% | 0% | 0% | 0.0 | 0.0 | 0.0 | 0.0 |
| MLP BC (D4RL architecture) | 76,041 | 1 | 0.0 | 0.0 | 0% | 0% | 0% | 0% | 0.0 | 0.0 | 0.6 | 1.2 |
| GRU, parameter-matched | 107,615 | 1 | 0.0 | 0.0 | 0% | 0% | 0% | 0% | 0.6 | 0.0 | 0.0 | 1.9 |

Lesions of the trained fly controller (same checkpoints, clean evaluation):

| Intact | edges_off | direct_only | deafferented_leg | head_sensory_deprived | state_reset_every_step |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

Zero action: 0.0.

## B1a: complete MaleCNS controller, FlyArm Panda tasks

### B1a pick-place (whole-brain-pick-place-001: complete; whole-brain-pick-place-002: complete)

| Controller | Params | Seeds | Grasp | Lift | Stable place |
|---|---|---|---|---|---|
| Full MaleCNS | 78,240 | 3 | 69/72 (24, 21, 24) | 58/72 (18, 18, 22) | 14/72 (5, 1, 8) |
| Full shuffled CNS | 78,240 | 3 | 65/72 (22, 22, 21) | 27/72 (10, 5, 12) | 10/72 (3, 0, 7) |
| GRU, parameter-matched | 78,368 | 3 | 43/72 (7, 16, 20) | 33/72 (7, 10, 16) | 4/72 (1, 0, 3) |

Post-training checks of the full-MaleCNS checkpoints (successes over all seeds):

| Ablation | grasp | lift | success |
|---|---|---|---|
| edges_off | 11/72 | 0/72 | 0/72 |
| direct_only | 58/72 | 37/72 | 4/72 |
| state_reset_every_step | 60/72 | 0/72 | 0/72 |

### B1a reach (whole-brain-reach-001: complete)

| Controller | Params | Seeds | Success | Mean final error (mm) |
|---|---|---|---|---|
| Full MaleCNS | 44,835 | 3 | 72/72 (24, 24, 24) | 4.65 |
| Full shuffled CNS | 44,835 | 3 | 72/72 (24, 24, 24) | 4.42 |
| GRU, parameter-matched | 45,139 | 3 | 72/72 (24, 24, 24) | 5.61 |

Post-training checks of the full-MaleCNS checkpoints (successes over all seeds):

| Ablation | success |
|---|---|
| edges_off | 0/72 |
| direct_only | 72/72 |
| state_reset_every_step | 66/72 |

