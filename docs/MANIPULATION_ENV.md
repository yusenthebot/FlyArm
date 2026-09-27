# Articulated multi-step manipulation environment

The Panda works a small tabletop kitchen: it pulls drawers open and pushes them shut, swings a cabinet door open and closed, puts scanned household objects into drawers, onto a shelf, into a bin or onto a marked region, and stacks them into towers.
Every task is a sequence of three to eight of these skills, and every skill has a strict physical success check.
The code is `src/flyarm/manipulation/`; it builds on the grasp task's arm, objects and physics backends (`src/flyarm/grasp/`), and the connectome controller trains on it by imitation and PPO (see Training on the connectome).

## Why a new package

`flyarm.grasp` is one object, one skill and one success rule, and it stays exactly as it was (its tests still pass).
The manipulation task adds furniture with joints, several objects in fixed slots, a task grammar, subgoal predicates, a sub-task cue and a different reward, which together are larger than the grasp task itself.
It therefore lives in its own package and reuses the grasp task through three narrow seams: `flyarm.grasp.arm` (the arm controller, IK and the mjbatch and MjData backends, moved out of `grasp.sim` without changing behaviour), `flyarm.grasp.scene.add_object` (the scanned objects and their contact sensors) and `flyarm.grasp.objects` (the catalog and its object split).

| Module | Contents |
| --- | --- |
| `furniture.py` | parametric furniture from MuJoCo primitives, its per-episode layout, the train and held-out ranges |
| `scene.py` | the MjSpec: Panda, furniture, objects, contact sensors, camera |
| `tasks.py` | skills, thresholds, task templates, receptacle fit rules, episode sampling |
| `sim.py` | `ManipulationSim`: readers, predicates, subgoal progress, observation, cue, reward, reset and step |
| `env.py` | `BatchedManipulation` (mjbatch) and `PandaManipulationEnv` (one MjData, rendering) |
| `splits.py` | the five preregistered splits and their committed record |
| `teacher.py` | `ManipulationTeacher`, the privileged scripted teacher |
| `rollout.py` | batched lockstep episodes for any controller: demonstrations, DAgger, evaluation, seed layout |
| `imitation.py` | behavior cloning and DAgger of the connectome controller (see Training on the connectome) |

## Scene

Four pieces of furniture share every episode, each with its own frame on the floor facing the robot.

- A drawer cabinet with one or two drawers side by side on slide joints (damping 4), each drawer a tray behind a front panel with a handle: a horizontal bar on two posts or a round knob on a stem.
  A drawer's travel is 85% of its depth.
- A cabinet with a hinged door over an interior shelf.
  The door is a lid hinged along the back edge (damping 0.5, opens to 1.66 rad, just past vertical, where gravity holds it open).
  Its handle is a bar on bearings that swivels about the hinge direction (so the pinch survives the lid turning under the hand) or a knob.
- An open bin with four low walls.
- A marked square region on the table, the base of towers.

Why the door is a lid: the 5-D action keeps the gripper pointing down, and with the hand 0.62 m in front of the base at shelf height, links 5 and 6 reach up to 0.63 m high, so reaching into a front-opening cabinet would need a 0.65 m tall opening.
A lid keeps the door physically necessary (the shelf cannot be reached while it is closed) and every motion top-down.

The drawer cabinet stands on one side of the robot and the cabinet on the other (50 to 62 degrees of azimuth, each side drawn per episode), the bin straight ahead behind the workspace and the region between them.
Two to four scanned objects from the split stand on the table (0.36 to 0.60 m from the base, within +/- 48 degrees, at least 10 cm from any furniture or the region so the open hand fits beside them, and clear of each other) or start inside a drawer.

Everything is built in MjSpec from primitives.
mjbatch copies one model for all simulations, so sizes, handle types and placement change per episode through per-simulation model fields (`geom_size`, `geom_pos`, `body_pos`, `body_quat`, `body_ipos`, `jnt_range`, `geom_contype`, `geom_conaffinity`, `geom_rgba`); an absent drawer or the unused handle type is made non-colliding and invisible.
MuJoCo's bounding volumes are compile-time, so the model is compiled at the largest configuration of every range and mid-phase collision is disabled; a test checks over random train and held-out configurations that every geom stays inside its compiled bounding sphere, geom box and body box.
Furniture and objects use stiff contacts (solref 0.004, solmix 1000), and the no-slip solver runs 4 iterations: without it a carried object crept out of the pinch over a hundred steps.

## Skills and success checks

Every check reads the simulator state after a control step.

| Skill | Effect |
| --- | --- |
| `open_drawer(d)` | slide position at least 75% of the drawer's travel |
| `close_drawer(d)` | slide position at most 4 mm |
| `open_door` | lid angle at least 1.58 rad |
| `close_door` | lid angle at most 0.02 rad |
| `pick(o)` | both fingers touch the object and it is out of its starting drawer (or 3 cm above its start) |
| `place(o, r)` | the whole object inside the receptacle's volume (4 mm tolerance), no robot contact, and at rest relative to the receptacle for 10 consecutive steps |
| `stack(o, b)` | the object's lowest point within 12 mm of the base's highest, its centre over the base's footprint, upright, untouched, both at rest, for 20 consecutive steps |

Whole object inside means 128 support points of the object's collision hull (its extreme vertex along 128 directions, within a millimetre of the hull) are inside; the eight corners of its bounding box overshoot a round object's hull by up to 40% of its radius and made an object resting against a drawer's front panel flicker out of the drawer.
Receptacle volumes: a drawer's tray interior (back wall to front panel) up to the carcass top, the cabinet interior above the shelf, the bin interior up to 0.3 m, and the region's square up to 0.3 m.
At rest means moving under 2 mm and turning under 0.03 rad per control step relative to the receptacle (displacements, not velocities: a light object on a drawer floor chatters at a few rad/s with sub-degree amplitude).
Once a placement or stack has held for its count it keeps holding while the geometry does, so a drawer carrying a placed object shut does not undo the placement.

## Tasks

A template is a list of (skill, object role, target) steps; binding it to an episode picks the drawer, the objects that fit every receptacle the template sends them to (a fit rule per receptacle with 12 mm margins, stackability and a stability rule for stacks), and zero or one distractor object (at least two objects in every scene).

| Template | Steps | Subgoals |
| --- | --- | --- |
| `put_away` | open drawer, place A in it, close it | 3 |
| `retrieve` | open drawer, pick A (starts in it), place A in the bin, close the drawer | 4 |
| `shelve` | open door, place A on the shelf, place B on the shelf, close door | 4 |
| `tower` | place A on the region, stack B on A, stack C on B | 3 |
| `sort` | place A in the bin, place B on the region, stack C on B | 3 |
| `tidy` | put A away in the drawer, then shelve B | 6 |
| `unpack` | retrieve A from the drawer to the region, then shelve B | 7 |
| `shelve_then_put_away` (held out) | shelve A, then put B away | 6 |
| `retrieve_to_shelf` (held out) | open door, open drawer, pick A, place A on the shelf, close drawer, close door | 6 |
| `tower_then_put_away` (held out) | place A on the region, stack B on A, then put C away | 5 |
| `full_cleanup` (held out) | put A away, place B on the region, stack C on B, shelve E | 8 |

The horizon is 200 + 300 steps per subgoal (1,100 to 2,600 control steps at 20 Hz).

Progress is derived from the scene every step (plus the rest counters of placements and stacks), never from a stored task pointer.
A subgoal is done when its effect holds, or when every later subgoal that needed it is done (an opened drawer that has since been closed after the placement it enabled counts as done).
The leading count is the number of subgoals done in order from the first; the current subgoal is the first one not done; the episode succeeds, and terminates, when all are done.
So undoing a subgoal (closing a drawer before the object is in it) makes it current again.

## Action

The grasp task's `[dx, dy, dz, dyaw, gripper]` in [-1, 1], 25 MuJoCo steps of 2 ms per action, with the same IK and posture control.

## Observation

220 floats (`OBS_DIM`; 217 before the imitation failure analysis added three), in fixed order (`flyarm.manipulation.sim.*_FIELDS`):

- robot, 21: joint positions and velocities, end-effector position, gripper yaw (sin, cos of twice the jaw angle), gripper opening, and the commanded yaw (dyaw integrates into it and the jaws follow with a lag);
- four object slots, 22 each: presence mask, position, position minus end effector, 6-D rotation, the grasp task's 5-D shape descriptor, linear velocity, grasped flag (zero when the slot is empty);
- furniture, 46: drawer cabinet xy and facing, drawers present, open fractions, both handle positions and their offsets from the end effector, handle type, cabinet xy and facing, lid angle, lid handle position and offset, lid handle type, shelf height, cabinet interior size, bin xy, facing and size, region xy and size;
- the cue, 65: the current subgoal's skill one-hot (7), which articulation (3), object slot one-hot (4), the object's position, offset from the end effector, rotation and descriptor, the receptacle one-hot (5 receptacles or "on an object"), the target point for the object's bottom and its offsets from the end effector and from the object, the receptacle's half size and facing, the handle position and its offset, the joint's open fraction, the direction the handle must move, the turn from the commanded jaw heading to the heading the subgoal needs to grasp (its handle or object) and to place (the receptacle's or base object's axis), computed by the same rule the teacher turns by, and progress (done over total, left over 8).

The cue is memoryless: it is recomputed from the scene each step through the same predicates as the reward, so a policy never needs to remember where in the task it is.
`cue=False` zeroes the whole cue block (the no-cue control); the rest of the observation is unchanged.
The privileged observation (233, for a critic) adds object masses, the per-subgoal done flags and the commanded yaw.

## Splits

Five splits, preregistered in `src/flyarm/grasp/catalog/manipulation_splits.json` (a test fails if the code and the record diverge).

| Split | Objects | Furniture | Templates | Seeds from |
| --- | --- | --- | --- | --- |
| `train` | 27 train objects | train ranges | 7 train | 0 |
| `iid_test` | train | train | train | 3,000,000 |
| `unseen_objects` | the grasp task's 14 held-out objects | train | train | 4,000,000 |
| `unseen_furniture` | train | held-out ranges | train | 5,000,000 |
| `unseen_composition` | train | train | 4 held out | 6,000,000 |

Held-out furniture draws every factor from a range disjoint from training: wider, deeper and taller drawers, drawer handles higher on the panel (0.6 to 0.7 of the panel instead of 0.35 to 0.6), a wider but shallower cabinet with a lower shelf and taller interior, the cabinets further round (62 to 68 degrees instead of 50 to 62) and turned further from facing the robot (10 to 16 degrees instead of 0 to 10), the drawer cabinet further and the cabinet nearer, a larger bin, and the knob instead of the bar on the lid.
Held-out compositions each contain an adjacent pair of steps no training template has (for example `close_door` then `open_drawer`, or `open_door` then `open_drawer`), and use only trained skills.

## Reward

Per step, with the defaults of `RewardConfig`:

- subgoal bonus 50 times the growth of the leading count past its high-water mark;
- shaping 10 times the change of the potential `Phi = leading + phi`, where `phi` in [0, 0.99] is the current subgoal's progress (for articulations 0.3 handle reach and 0.7 joint fraction; for picks reach, grasp and effect; for place and stack reach or grasp, grasp, and closeness of the object's bottom to its target point);
- disturbance 5 times the change of minus the displacement of objects and articulations the task does not involve (a potential too);
- stray contact -0.05 when the robot touches anything the current subgoal does not need;
- action cost -0.01 mean(a^2).

Invariants (`RewardConfig` checks the first two):

- Every per-step level term is a penalty, so the stalling floor, the discounted value of holding the best non-terminal state forever, is 0, and the bonus must exceed it.
- The bonus exceeds the shaping weight, which is the most shaping can pay for one subgoal, so completing a subgoal is always worth more than approaching it.
- Shaping and disturbance are potential differences, so they telescope over an episode and cannot be farmed by oscillating; undoing a subgoal costs exactly what redoing it pays back.
- The bonus is paid once per subgoal, in task order, only when the leading count passes its high-water mark: undoing and redoing a subgoal pays no second bonus.

The reward shapes training only; evaluation reports the success rule.

## Scripted teacher

`ManipulationTeacher(sim).act()` returns an action for every environment from privileged state (object poses and sizes, handle positions, joint positions, contacts).
Each step it takes the current subgoal from the scene and runs that skill's controller:

- articulation: travel above everything, descend beside the handle, centre on it using finger contacts, pinch (the lid bar front to back, knobs along the hinge), move the handle along its joint path to a goal past the stop (so the servos' lag does not leave it a few millimetres short; drawers open all the way, so brushing one while picking from it cannot push it back under the threshold), release and step back;
- pick, place and stack: the grasp task's rule (jaws across the narrow side, or along a drawer's width when picking from one and that is still across the object, so no finger lands on the front panel; pads at the geometry-derived height, proportional xy alignment (an integral term was removed with the gravity compensation, see Imitation failure analysis), closes once the object lies between the pads with 3 mm to spare, at most 5 mm off centre along the jaws, see Skill-level DAgger failure analysis), carry above every furniture top and object, turn to the receptacle's axis (finishing the turn before coming within 15 cm of the target, so the hand does not sweep an opened lid shut), lower until the bottom reaches the target surface or the descent stalls, release and rise.

Its memory is the subgoal it is executing, a phase, a counter and a held xy; watchdogs restart a subgoal whose phase stalls.
It re-derives its phase from the scene whenever the current subgoal changes, so it can label any state a learner reaches (DAgger), and it never writes simulator state.

### Teacher table

Successes over 20 episodes per cell, all cells of a split run in one batched environment on that split's own seed block (`docs/results/manipulation-teacher.json`, with mean steps to success and the subgoal each failure was stuck at).

Measured on the task as changed by the imitation failure analysis (gravity-compensated arm with armature 0.3, teacher without its integral term, travel sink and lid leads, see there); the value before those changes, on the same seeds, is in brackets where it differs.

| Task | train | iid test | unseen objects | unseen furniture | unseen composition |
| --- | --- | --- | --- | --- | --- |
| `put_away` | 20/20 | 20/20 | 20/20 | 20/20 |  |
| `retrieve` | 20/20 | 20/20 | 20/20 (19) | 20/20 |  |
| `shelve` | 19/20 | 19/20 (20) | 20/20 (18) | 16/20 |  |
| `tower` | 20/20 | 20/20 | 20/20 | 20/20 |  |
| `sort` | 20/20 | 20/20 | 20/20 (14) | 20/20 (19) |  |
| `tidy` | 20/20 | 20/20 (18) | 20/20 (19) | 19/20 (17) |  |
| `unpack` | 20/20 | 20/20 (19) | 16/20 (17) | 18/20 |  |
| `shelve_then_put_away` |  |  |  |  | 20/20 (18) |
| `retrieve_to_shelf` |  |  |  |  | 15/20 |
| `tower_then_put_away` |  |  |  |  | 20/20 |
| `full_cleanup` |  |  |  |  | 17/20 |

Every train cell is at 90% or above (139/140, as before); every iid cell too; over all cells 619 successes against 603.
No cell lost more than one success (two lost one: `shelve` on iid test and `unpack` on unseen objects), and the largest gain is `sort` on unseen objects, 14 to 20: without the sag, tops no longer tip off their base on release.
Episodes take 2 to 7% more steps on the train and iid splits (the hand no longer sinks toward the next target by itself between moves) and up to 20% fewer on unseen furniture.
Before those changes, with the redundant contact pairs excluded, every cell was within one success of the table before the exclusion (train 139 against 138).
Cells under 90%, all on held-out splits:

- `unpack` on unseen objects (16/20): three failures at the placement on the region and one at the pick from the drawer; not diagnosed further.
- `shelve` on unseen furniture (16/20): the held-out cabinet (knob on the lid, turned further from the robot, lower shelf); all four failures are at a shelf placement.
- `retrieve_to_shelf` (15/20): the only template that works a drawer while the lid stands open; all five failures are at the pick from the drawer (a finger on the front panel pushing the drawer back under its threshold, reduced but not removed by picking with the jaws along the drawer's width).
- `full_cleanup` (17/20): each failure is stuck at a placement; not diagnosed further.

The teacher's episodes are what imitation would learn from, so a teacher failure is also a state the learner will not see solved; DAgger labels still come from the same controllers.

## Throughput

128 batched environments on the train split, 4 simulation threads (`FLYARM_SIM_THREADS=4`), Apple M5 Max: 3,040 environment steps per second with random actions and 2,200 with the teacher in the loop (`docs/results/manipulation-teacher.json`, `throughput`), against 2,330 and 1,820 before the imitation fixes and 1,350 and 1,130 before the contact exclusion.
Physics is 75 to 85% of a step (25 MuJoCo steps over 43 furniture geoms, the no-slip solver and up to four object meshes); the task rules, observation and reward cost 10 to 16 ms per step for all 128 environments, 5 ms of it the hull containment test.
That is five times slower than the grasp task per environment step, and the horizons are 5 to 13 times longer.

## Excluded contact pairs

Two body pairs never collide (`EXCLUDED_PAIRS` in `scene.py`, MjSpec contact exclusions): the lid and the cabinet, and the left and the right finger.
Both are already enforced by joint limits: the hinge's lower limit holds a closed lid at 0 on the cabinet walls, and the finger joints' lower limit stops closing fingers at touching.
Before the exclusion they were most of the scene's contacts: a closed lid made 20 contacts in every episode (the cabinet is welded to the world, so MuJoCo's parent-child filter does not apply), and fingers closed on nothing made up to 44.
Measured effect at 128 environments: the environment alone went from 1,350 to 2,330 steps/s with random actions; profiled over steps 300 to 400 of an undertrained connectome policy, a step's physics went from 264 to 176 ms; PPO rollouts went from 555 to 820 steps/s in the first iteration and from 410 to 450 to 550 to 580 afterwards.
Without the walls a closed lid rests on its hinge limit alone, whose soft constraint lets it sit 1.5 mrad past 0 (0.3 mm at its front edge), far inside the 0.02 rad closed threshold.
The teacher table barely moved (above), and the single-versus-batched equivalence test still holds under 1e-6.
What remains in a contact-heavy step is real contact (fingers pressed into the table or a lid, objects on the floor, 5 to 43 contacts per environment) and the no-slip solver that scales with it.

## Reproduce

```
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/manipulation_smoke.py
FLYARM_MODEL=assets/menagerie/franka_emika_panda/scene.xml PYTHONPATH=src .venv/bin/python -m pytest tests/test_manipulation_env.py tests/test_manipulation_tasks.py
```

The smoke script needs the grasp objects (`scripts/fetch_grasp_objects.py`).
It writes the per-task per-split teacher table, the throughput and the video record to `docs/results/manipulation-teacher.json`, and a labelled video (task, split, step, subgoals done and the current subgoal on every frame) of the teacher on every task template, one unseen-objects episode and one unseen-furniture episode to `runs/manipulation/teacher-smoke.mp4`.
After changing templates or ranges, `PYTHONPATH=src .venv/bin/python -c "from flyarm.manipulation.splits import write_record; write_record()"` rewrites the split record; that is a change to the preregistration and belongs in its own commit.

The tests cover split disjointness and the committed record, every sampled episode fitting its receptacles, each skill's success check on constructed states (open and closed thresholds on either side, an object placed at rest versus not yet at rest versus outside, a stack before and after 20 stable steps versus beside the base), the reward floor, one bonus per subgoal and telescoping shaping, the cue and the no-cue control, step-for-step equality of the batched and single environments over recorded teacher actions (drawer and door tasks, under 1e-6), furniture bounds, the hull points used for containment, and the teacher completing drawer tasks.

## Known limits

- The door is a top-hinged lid, not a side-hinged front door (see Scene).
- Everything the robot manipulates is a primitive; furniture has no visual meshes.
- Held-out furniture changes the drawer handle's height but not its type; only the lid's handle type is held out (the knob).
- Objects keep the grasp task's limits: convex hulls, toy-scaled, one density, one friction.
- Stacks are limited to boxes, cans and bowls under 9 cm, with the stability rule of `tasks.stable_on` (the top no larger, longer or much heavier than the base, and no taller than 1.5 times its narrow width); the tower's base is the largest footprint.
- A drawer's placement point is the middle of the part a drawer opened to the threshold leaves outside the carcass, so objects longer than that part cannot go in drawers and the fit rule excludes them.
- The teacher's rates are on 20 episodes per cell; a cell at 85% could land a few points either way with other seeds.

## Training on the connectome

The controller is the B1a whole-body interface: the 220 observation features go through a trainable linear encoder into the 1,846 ascending neurons of the frozen MaleCNS (166,700 neurons), the 1,314 descending and 708 VNC motor neurons are read out through a frozen unit-norm calibration (research log E26) and a trainable linear decoder gives the 5 actions, one per control step.
Only the two linear maps train; the connectome never changes.
The recipe is the kitchen's (docs/MILESTONE_KITCHEN.md): imitation, then PPO with a demonstration term, a bonus paid in task order, potential shaping and motion-quality terms.

```
PYTHONPATH=src .venv/bin/python -m flyarm.cli manipulation imitate --config configs/whole-brain-manipulation.json --output runs/whole-brain-manipulation-001
PYTHONPATH=src .venv/bin/python -m flyarm.cli rl manipulation --config configs/ppo-manipulation.json --output runs/ppo-manipulation-001
PYTHONPATH=src .venv/bin/python -m flyarm.cli rl watch --run runs/ppo-manipulation-001
```

All three need `FLYARM_SIM_THREADS=4`, the connectome pack at `data/whole_brain/malecns-v1.0-c3` (`--pack` otherwise) and the grasp objects in `assets/objects`.
The PPO config's `base_run` names the imitation run.

### Imitation (flyarm.manipulation.imitation)

A sibling of flyarm.whole_brain.experiment and flyarm.flyleg.experiment rather than a branch of them: both step one Gymnasium episode at a time, which suits 100 to 400-step episodes, while here episodes last 1,100 to 2,600 steps and every number is a per-split, per-template rate.
So demonstrations, DAgger, selection and evaluation run batched and in lockstep across environments (flyarm.manipulation.rollout): the connectome is called once per control step for every episode of every split.
The policy classes, the sequence trainer (truncated BPTT through the frozen connectome), the interface, the shuffle and the phase-selection rule are the existing ones.

1. Teacher demonstrations: `train_episodes_per_template` episodes of each of the 7 training templates (24, so 168 episodes and about 175,000 steps), recorded with the skill of the current subgoal at every step.
2. Readout calibration (unit norm) and observation normalization on those steps.
3. Behavior cloning: L1 loss, per-step weights that give every skill the same total weight (capped at 5 times the mean), window sampling (every update fits 32 windows of 16 steps drawn from all demonstrated steps, each after a 16-step burn-in without gradient; the earlier episode sweep is `sampling: episodes`), 2 decoder-only warm-up epochs, joint and object velocities removed from the policy's input (`velocities: false`).
   Episodes stay in host memory and each batch is copied to the GPU when it is used.
4. DAgger rounds: the learner drives `dagger_episodes_per_template` new episodes per template, the teacher labels every visited state (it re-derives its phase from the scene), the teacher's action is executed instead with probability 0.5 in round 1, halving each round (four rounds in the shipped configs), and training continues on the aggregate.
5. Closed-loop selection on validation episodes of the train split: within behavior cloning every `select_every` epochs, and among the phases at the end (research log E33); success rate first, the mean fraction of subgoals done breaks ties.
6. Evaluation of the selected checkpoint and of the teacher on the same test episodes of every held-out split: success, subgoals completed, mean steps to success, per template, and the motion-quality metrics (stray-contact fraction, mean |a|, saturated fraction, mean |change of a|, final disturbance).

The encoder trains at a tenth of the decoder's learning rate (`encoder_learning_rate_scale`).
Measured on 21 demonstration episodes, three epochs from the same calibrated start (L1 by quarter of epoch 3; input-current RMS into the ascending neurons after epoch 1):

| Encoder | Epoch 3 L1 | Input RMS |
| --- | --- | --- |
| frozen (decoder only) | 0.26 to 0.27 | 0.48 |
| same rate as the decoder | 0.25 to 0.30 | 2.15 |
| a tenth of it | 0.23 to 0.26 | 0.51 |

At the full rate, Adam's early steps move all 217 x 1,846 encoder weights (220 x 1,846 now) by about the learning rate on correlated inputs and drive the ascending neurons into tanh saturation within one epoch, the input-side counterpart of the readout effect in E26; the kitchen's whole-body interface had 30 inputs and did not show it.
The controls (`policies`: `shuffled`, `gru`, `mlp`) run through the same code; their input modules train at the full rate.

Seeds (flyarm.manipulation.rollout): episode i of a split's template t has seed `seed_start + offset + 1000 t + i`; demonstrations use train offset 0, validation 100,000, DAgger round r 200,000 + 10,000 r, PPO training episodes start at 1,000,000 (below 3,000,000, where the first held-out block starts), and test episodes use each held-out split's own block.
Adding episodes never changes existing ones.

### PPO (flyarm.rl.ppo_manipulation)

`ManipulationTask` is the `TaskAdapter` for the unchanged `train_ppo`; the trainer gained two optional hooks (per-episode horizons for the critic's time feature, and a selection key).

- Reward: the environment's RewardConfig built from the config: subgoal bonus 50 paid once and in task order, potential shaping at weight 10, stray contact 0.05, disturbance 5, action 0.01, smoothness 0 (the kitchen used it; it is off by default here until a run shows commands need it).
  Every level term is a penalty, so the stalling floor is 0 at any gamma and the config's check recomputes it for its own gamma (default 0.995: the next subgoal, about 300 steps away, is discounted to 0.22 instead of 0.05 at 0.99).
- Critic: the 233-feature privileged observation plus the step over the episode's own horizon.
- DAPG: the squared error to the imitation run's own teacher demonstrations (train.npz), at most `bc_max_steps` steps drawn with the skill-balanced weights, replayed once through the frozen encoder and connectome from a zero state; it needs the encoder frozen (`encoder_lr` 0).
- Warm start from the imitation checkpoint; `init_checkpoint` continues from a PPO checkpoint (the critic starts fresh, so `critic_warmup` stays above 0; 50 by default, E39, E46).
- Advantage clip 10 (E39), exploration log std -1.2.
- Evaluation every `eval_every` iterations: `eval_episodes_per_template` test episodes of every split in `eval_splits` and `val_episodes_per_template` validation episodes of the train split, all in lockstep; checkpoints are selected on validation only.
- Training curves carry `subgoals_per_episode` (mean over finished episodes) and the batch peak `max_subgoals_in_one_episode`.
- `flyarm rl watch` re-records one clip of the newest checkpoint (two train-split episodes from seed 60,000, templates by seed) into `progress/latest.mp4`, as for the kitchen.

### Curriculum PPO (flyarm.manipulation.curriculum)

Modelled on what solved the kitchen: resets to demonstration states (E53), the bonus paid in task order (E47), potential shaping (E48) and the DAPG term (E46); still only the two linear maps are trained.

- Subgoal resets: the teacher runs `bank_episodes_per_template` episodes per template of the train split (seeds from 400,000) and the full state at the start of every subgoal is stored (`SubgoalBank`, saved as bank.npz in the run).
  `reset_to_subgoal` rebuilds the episode from its seed and template (a fingerprint of the episode guards against drift) and restores the physics state, the arm controller's state, the task's counters and baselines, and the derived fields the controller reads after `mj_step`, so the replay is bit for bit (test: a restored state replayed with the recorded actions matches for 60 steps, difference 0.0).
  The first k subgoals count as done and pay no bonus, the episode succeeds after its budget of subgoals (or the template's end) and is cut at 300 steps per subgoal it has to do; the cue, the ordered bonus, the shaping target and the stray-contact allowances all follow from the preset.
  Only states are stored, never actions.
- Curriculum (`curriculum` in `ManipulationPPOConfig`, stages switched by iteration budget through an optional `train_ppo` hook): each training episode is a true start with the stage's share, otherwise a subgoal start with skills equally likely and a budget drawn from the stage's range.
  configs/ppo-manipulation-curriculum.json: 300 iterations of single subgoals (20% true starts), 300 of two or three (25%), 400 of full templates from true starts only; everything else as configs/ppo-manipulation.json (gamma 0.995, DAPG 1.0, advantage clip 10, critic warm-up 50, the motion-quality terms), base run runs/whole-brain-manipulation-002.
  One gamma for all stages: a single subgoal takes 100 to 350 steps, which 0.995 covers as well as 0.99.
- Evaluation: every split from true starts (the headline), validation from true starts (selection), and single-subgoal success per skill from a separate validation bank (seeds from 500,000, `skill_eval_episodes` per skill), reported as `skill:<name>`.
  Training curves carry the stage, the counts of subgoal and true starts, and subgoals per episode counting only those the episode earned.

Run (after the imitation run): `FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli rl manipulation --config configs/ppo-manipulation-curriculum.json --output runs/ppo-manipulation-curriculum-001`.

Smoke (a tiny imitation base, 128 environments, 6 iterations through both stage switches, bank of 3 episodes per template: 86 subgoal starts over all 7 skills, validation bank 30): 8.4 minutes end to end; rollouts at 530 to 645 steps/s with the full imitation run sharing the GPU, 14.5 s per iteration including the update; the per-skill and per-split evaluations ran and the checkpoint was selected on validation.
Estimate for the shipped config on the shared GPU: 1,000 iterations x 14.5 s = 4.0 hours, 11 evaluations x 4 to 6 minutes = 45 to 65 minutes, the banks (168 teacher episodes) and the DAPG replay about 8 minutes: about 5 hours, less with the GPU to itself.

### Sensory encoder (branch feat/sensory-encoder)

Curriculum PPO on the linear interface did not learn (research log), so the peripheral interface gains a trainable nonlinear sensory encoder: `BrainPolicy(encoder="mlp")` maps the observation through hidden layers (default 2 x 256, tanh) and a linear layer to the 1,846 ascending currents; the connectome stays frozen and the readout linear, so the claim becomes "only the peripheral interface is trained".
`encoder="linear"` is the default and bit-identical to every earlier policy (test: same weights, same parameter names).
At the start the output layer is scaled so the ascending currents have the linear encoder's RMS (0.5); the encoder then trains through BPTT in imitation, through `encoder_pass` in PPO (any module works), in ablation clones and in checkpoints.
In PPO the encoder stays frozen whenever the DAPG term is on (the config refuses both): the demonstration features are then exact, imitation has already trained the encoder with full BPTT gradients where PPO's encoder pass has one-step ones, and the readout is what PPO tuned on the kitchen.

Capacity probe (`scripts/sensory_encoder_probe.py`, docs/results/sensory-encoder-probe.json): the 28 demonstrations and held-out split of the rate-response probe, every variant trained by the imitation pipeline's trainer for 20 epochs (window sampling, L1, the epoch with the lowest training loss kept, so the held-out episodes select nothing), controls matched to the encoder policy's 606,905 trainable parameters.

| Variant | Trainable parameters | L1 train / held out |
| --- | --- | --- |
| MLP encoder + measured connectome + linear readout, encoder learning rate x1 | 606,905 | 0.134 / 0.166 |
| the same, encoder learning rate x0.1 | 606,905 | 0.189 / 0.218 |
| MLP encoder + degree-preserving shuffle | 606,905 | 0.132 / 0.159 |
| MLP encoder + direct input-to-output synapses only | 606,905 | 0.145 / 0.174 |
| linear encoder + measured connectome (the pipeline's x0.1) | 418,081 | 0.216 / 0.236 |
| MLP policy, 2 x 674 | 607,279 | 0.100 / 0.145 |
| GRU, 352 units, 128-step burn-in | 607,205 | 0.099 / 0.135 |
| GRU, 16-step burn-in | 607,205 | 0.261 / 0.273 |
| GRU, episode sweep | 607,205 | 0.281 / 0.299 |
| frame-wise references: linear policy / MLP 3 x 512 | | 0.211 / 0.267, 0.021 / 0.140 |

Reading: the encoder variant passes the bar (0.166 held out, well under 0.22 and near the matched MLP's 0.145), so it earns a full run; but the connectome contributes nothing specific to it.
The degree-preserving shuffle fits as well (0.159) and the direct synapses alone almost as well (0.174), so the gain is the encoder's, and the graph acts as a fixed random projection plus the ascending neurons' saturation (the trained encoder drives their currents to RMS 6.1, far into the tanh, from 0.5).
The encoder needs its full learning rate; the tenth that keeps a linear encoder out of saturation leaves this one underfit.
The GRU control needs a burn-in longer than its memory: with 16 steps its windows fit (0.09) but whole episodes from a zero state do not (0.26), and the episode sweep underfits it; the connectome's memory is about 0.2 s, so it keeps 16.

Configs (connectome seed 0, the fixed imitation recipe plus the encoder, encoder learning rate x1): configs/whole-brain-manipulation-encoder.json (measured connectome), -encoder-shuffled.json (the same encoder on the degree-preserving shuffle), -encoder-baselines.json (parameter-matched MLP and GRU, burn-in 128), and configs/ppo-manipulation-encoder-curriculum.json (the curriculum PPO from runs/whole-brain-manipulation-encoder-001, encoder frozen, DAPG on).
Smoke (tiny imitation with the MLP encoder, one DAgger round, then 6 curriculum PPO iterations through both stage switches and a progress clip): imitation 3.7 minutes, PPO 3.7 minutes at 1,160 to 1,325 steps/s with the GPU free.
Estimates with the GPU to one run at a time: encoder imitation about 4.5 hours (the linear run took 4.0; an encoder epoch costs about 15% more), the shuffle the same, the MLP and GRU baselines about 2 hours (their training is minutes, rollouts and evaluations dominate), curriculum PPO about 3 hours (1,000 iterations of about 8 s plus 11 evaluations).

### Skill-level DAgger (flyarm.manipulation.skill_dagger)

The encoder imitation run (runs/whole-brain-manipulation-encoder-001) fit the teacher to the MLP's level (L1 0.086 to 0.093 train, 0.094 validation) and still completed no episode on any split (subgoal fraction 0.060 iid, 0.004 unseen composition).
That is covariate shift: four DAgger rounds of 56 full-length episodes are far too little on-policy data for tasks of 1,100 to 2,600 steps.
The teacher is Markov and cheap and the batched environment is fast, so skill-level DAgger scales the on-policy data by an order of magnitude and starts on short horizons.

- Round 0 records `teacher_episodes` teacher episodes.
  Round r >= 1 rolls out the current learner on `episodes_per_round` episodes in the batched environment, executing the teacher's action instead with probability `betas[r - 1]` (0.5, 0.25, then 0).
  Every visited state is labelled with the teacher's action (test: a replay of the learner's actions with a watching teacher gives the same states and labels for 200 steps).
- Episodes follow the round's stage (`stages`, switched by round): a share of true starts running the whole template (always at least one), the rest from the curriculum's subgoal bank (fingerprint and bit-exact restore), skills equally likely, each with a budget of subgoals from the stage's range.
  Default: 4 rounds of single subgoals (10% true starts), 3 rounds of two or three (15%), 3 rounds of full templates from true starts only.
- The aggregate lives on the host with episodes stored end to end (`StepData`, no padding), weighted so every skill has the same total weight (capped at `max_skill_weight`).
- Each round trains the warm-started controller for a fixed number of window-sampled updates (`train_windows`: 32 windows of 16 steps after the config's burn-in, windows never cross episodes, fresh Adam per round), 3,000 in round 0 (the first 300 decoder only for the connectome) and 1,500 afterwards.
  Round 0 sets the frozen normalization, the encoder scale and the readout calibration from the teacher's data, as imitation does.
- Per round `{kind}-{seed}/round-XX/` holds the round's labelled steps (data.npz), the weights and metrics.json: rollout statistics (steps, labelled steps per second, share of teacher steps), the training loss, single-subgoal success per skill from the validation bank, and success, subgoal fraction and selection score from true starts on the train split's validation seeds.
  metrics.json is written last, so `--resume` continues after the last complete round (a run that finished a controller skips it).
- The best round on validation (success first, subgoals second, mean skill success third, later rounds on ties) is evaluated on every split and every skill with the quality metrics.
- The run directory has an imitation run's layout (config.json is the controller's `ManipulationImitationConfig`, interface.json, train.npz with the round-0 teacher episodes, `{kind}-{seed}/policy.safetensors`), so curriculum PPO warm-starts from it unchanged.

Configs (seed 0, the encoder imitation recipe: MLP encoder, encoder learning rate x1, L1, window sampling): configs/skill-dagger-connectome.json, -shuffled.json (degree-preserving shuffle), -mlp.json and -gru.json (matched to the MLP-encoder connectome's 606,905 parameters, so they keep `encoder: mlp`; the GRU keeps its 128-step burn-in), 256 episodes per round, 10 rounds, 48 hour limit.
configs/ppo-manipulation-skill-dagger.json is the encoder curriculum PPO from runs/skill-dagger-connectome-001 starting at stage 2 (300 iterations of two or three subgoals, 400 of full templates, evaluation every 50).

```
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli manipulation skill-dagger --config configs/skill-dagger-connectome.json --output runs/skill-dagger-connectome-001
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli manipulation skill-dagger --config configs/skill-dagger-connectome.json --output runs/skill-dagger-connectome-001 --resume
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli rl manipulation --config configs/ppo-manipulation-skill-dagger.json --output runs/ppo-manipulation-skill-dagger-001
```

Smoke (connectome with the MLP encoder, 64 episodes per round, one round of single subgoals and one of two or three, 300 and 150 updates, banks of 2 and 1 episodes per template), the encoder PPO and the matched imitation sharing the GPU: 17.7 minutes end to end, peak memory 14.4 GB.
Round 0: the teacher labelled 16,733 steps in 55 s (true starts 6 of 6, subgoal starts 50 of 58); L1 0.39 to 0.23 in 161 s (0.54 s per update).
Round 1 (beta 0.5): 45,604 labelled steps in 190 s (240 per second, 49% teacher steps), aggregate 62,337; L1 0.28 on the aggregate after 150 updates (87 s).
Each round's evaluation (7 true starts, 14 single subgoals) took about 2 minutes; validation and every split were 0 successes at this size, as expected; `--resume` on the finished run returned its evaluation in 9 s.
Window updates of the matched controls on that aggregate: MLP 11.7 ms, GRU with 128-step burn-in 83 ms.
Curriculum PPO from that run (16 environments, one iteration per stage from stage 2, DAPG on 2,000 steps of its train.npz) loaded the selected checkpoint, scored it on every split and skill, and completed.

Estimates for the shipped configs on the shared GPU: about 2.0 to 2.6 M labelled steps (stage 1 rounds about 120,000, stage 2 about 180,000, stage 3 about 430,000 when the learner fails and runs to the horizon).
Connectome or shuffle: 16,500 updates x 0.55 s = 2.5 hours, 9 learner rounds of rollouts 1 to 1.5 hours (a lockstep round lasts as long as its longest episode), 10 evaluations about 30 minutes, banks and final evaluation about 20 minutes: about 4.5 to 5 hours each, less with the GPU free.
MLP about 1.5 to 2 hours and GRU about 2 to 2.5 hours (training 3 and 25 minutes, rollouts and evaluations dominate).
The PPO from stage 2: 700 iterations x 10 to 15 s plus 15 evaluations, about 3 to 4 hours.
Host memory grows with the aggregate (2.6 M steps are 2.3 GB, held twice while a round concatenates), so run the four controllers one or two at a time.

### Skill-level DAgger failure analysis (branch fix/skill-dagger)

runs/skill-dagger-connectome-001 (10 rounds, 2.0 M labelled steps) peaked at round 1 (beta 0.5: mean single-subgoal success 0.29 over the seven skills) and fell to about 0 once the learner drove alone (beta 0 from round 3), while its aggregate L1 rose with data (0.092, 0.114, 0.131, then 0.13 to 0.14).
Same failure as the imitation runs, so one hypothesis at a time, each falsified on a real closed loop.
Tool: `scripts/skill_dagger_diagnostics.py` (single-subgoal episodes from the run's validation bank, 12 per skill, 8 for pick, 84 in all, the teacher labelling every state the controller visits; rows in docs/results/skill-dagger-diagnostics.json).
Failure reasons come from the subgoal's own shaping term: `timeout` (progressed, ran out of steps), `stalled`, `wrong_direction`, `lost_grasp`.

H-A, the learner's commands are uniformly too small (a regressor shrinking a switching teacher): refuted as stated.
Scaling the round-1 connectome's actions at evaluation does not help: mean skill success 0.286, 0.280 and 0.226 at gains 1.0, 1.5 and 2.0, and failures are timeouts (51 of 60 at gain 1), not wrong directions (1) or lost grasps (0).
On the teacher's own states (the controller watching the teacher drive) the commands match the labels: norm ratio 0.99 and cosine 0.997 where the teacher is at full speed, slopes on the label 0.83, 0.95 and 0.93 for x, y and z, L1 0.09 to 0.12.
Only yaw is shrunk there (slope 0.34).
On the controller's own states the commands are small and unrelated to the labels: mean |a| 0.16, 0.31, 0.41 and 0.06 for x, y, z and yaw against the labels' 0.40, 0.44, 0.78 and 0.79, slopes 0.16, 0.45, -0.03 and 0.06, cosine -0.14 at full-speed labels.
So the shrinkage is a symptom of where the learner is, not a property of the regression.

H-B, the teacher's labels flip between neighbouring states along learner trajectories: refuted.
The fraction of consecutive steps where a dimension's label flips sign (both sides above 0.05) or jumps by more than 0.5 is 2.7% along the controller's trajectories and 3.6% along the teacher's (x, y and z alone: 2.5% both).

H-C, the connectome's state starts at zero at a subgoal reset: refuted.
With 16 teacher-driven steps before the controller takes over, mean skill success is 0.298 with the state kept and 0.286 with it reset at the handover, against 0.286 without the burn-in.

H-D, a fixed number of updates per round spreads over a growing aggregate (0.4 epochs a round by 2 M steps), so the learner underfits its own states: refuted as the cause.
The matched MLP through the same pipeline with 10 epochs per round (`epochs_per_round`) fits better (L1 0.057 and 0.070 in rounds 0 and 1 against the connectome's 0.092 and 0.114) and collapses the same way: mean skill success 0.14 and 0.15 at beta 1 and 0.5, then 0.01, 0.00, 0.06 and 0.04 with beta 0, L1 jumping to 0.13 as the learner's own states arrive.

H-E, the learner stalls a few millimetres short of the teacher's gates: supported.
The teacher is a proportional controller at one full action per 14 mm (0.05 rad in yaw) with phase gates at 8 mm and 0.05 rad (hover to descent) and 2 to 3 mm (descent to closing).
The observation carries the offsets it acts on, normalized by their spread over whole episodes (about 0.1 m).
- Round 1 of the connectome: along its own trajectories the teacher is in APPROACH 88% of the time, never passing the hover gate; its yaw labels are saturated 73% of the time while the controller's yaw command averages 0.06 (the heading error is in the observation and the approach yaw label equals clip(error / 0.05) to 0.004).
- Round 9: the controller is stationary 54% of its steps (the hand moved under 5 mm over 10 steps), 85% of them with the teacher in DESCEND at the hover point 7 cm above the handle, labelling z -0.6 and an xy correction implying 3.2 mm (quartiles 1.9 to 4.9 mm) of residual error; the controller's command there is |z| 0.08 and |x|, |y| 0.01.
- The labels are a learnable function of the observation: a frame-wise MLP of the matched size fits held-out episodes of the learner's states with sign agreement 0.94 to 0.98 wherever |label| > 0.5.
  But it resolves small commands poorly (held-out L1 0.27 on the learner's states for x, y and z).
- Control-scale features (tanh of every hand-relative offset at 1 and 4 cm and of the two heading errors at 0.05 and 0.2 rad, appended inside the policy) cut the frame-wise held-out L1 from 0.182 to 0.126 (the learner's states: 0.273 to 0.186; labels between 0.1 and 0.6: sign agreement 0.83 to 0.91).
- Closed loop, frame-wise MLP DAgger with the same banks and round plans (256 episodes a round, about 10 epochs a round): without the features mean skill success 0.14, 0.24, then 0.16, 0.07, 0.04 and 0.08 with beta 0; with them 0.20, 0.42, then 0.48, 0.41, 0.41 and 0.48, L1 staying at 0.043 to 0.053 instead of rising to 0.08.

H-F, spurious stops where the teacher's rules switch on hidden history: supported, partly fixed.
With the features the controller reaches the handle, and then stalls again once it drives alone: the fixed connectome below is stationary 20% of its steps after round 1 and 46% after round 3, most of it (56%) with the teacher in APPROACH while the hand is already at the handle or object (1 to 4 mm away in xy, 13 to 18 cm below the transit height), labelling "rise to the hover" (z +1) where the teacher's own episodes, which always pass the hover's strict alignment gate on the way down, label "descend" or "close".
The same state gets opposite labels depending on a path the observation cannot show, and the regression settles between them.
Fix at the root in the teacher: an articulation's APPROACH re-enters the descent when the hand is already below the hover, within 12 mm of the handle, turned within 0.15 rad and open (`REENTRY_XY`, `REENTRY_YAW`), so the label there no longer depends on the path.
The teacher never meets that condition on its own trajectories: the teacher table is unchanged cell for cell (619 of 640, identical mean steps), and so are the banks and demonstrations.
Objects keep the old rule: rising from a just-placed object is how the teacher lets a placement settle, and a re-entry there picks it up again (tried: `retrieve` and `unpack` fell to 0 of 20).
Frame-wise probe with the features and the re-entry: mean skill success 0.23 and 0.45 at beta 1 and 0.5, then 0.51, 0.44, 0.54 and 0.53 with beta 0 (0.48, 0.41, 0.41 and 0.48 without the re-entry).

The fix in the pipeline:
- `control_features` (flyarm.manipulation.features, a fixed expansion inside the policy; the environment and its observation are unchanged; matched budgets count the extra inputs);
- `epochs_per_round` (each round trains that many passes over the aggregate, capped by `max_updates_per_round`) and 64 windows an update (the connectome's update costs 0.98 s at 64 windows against 0.92 s at 32 on the shared GPU);
- windows drawn so that every step is covered equally often (`window_starts`; drawing the first step uniformly covered an episode's first 15 steps up to 16 times less, and those are where a reset episode starts moving);
- the teacher's memoryless re-entry at handles.

Pipeline results (single subgoals from resets, round 0 the teacher, round 1 beta 0.5, then beta 0; mean single-subgoal success over the seven skills, 12 validation-bank episodes per skill, learner alone):

| Run | Episodes a round | Round 0 | Round 1 | Rounds with beta 0 |
| --- | --- | --- | --- | --- |
| Connectome, runs/skill-dagger-connectome-001 (before; beta 0.25 in round 2: 0.21) | 256 | 0.04 | 0.29 | 0.04, 0.04, 0.07, 0.07, then 0 to 0.07 on longer episodes |
| Matched MLP, 10 epochs a round, no features | 256 | 0.14 | 0.15 | 0.01, 0.00, 0.06, 0.04 |
| Matched MLP, features, 1,500 updates a round | 256 | 0.24 | 0.30 | 0.08 (stopped) |
| Matched MLP, features, 10 epochs a round | 256 | 0.24 | 0.27 | 0.14, 0.23, 0.31, 0.33 |
| Matched MLP, features, 10 epochs, 64 windows, even coverage | 256 | 0.25 | 0.29 | 0.32, 0.23, 0.20, 0.17 |
| Connectome (MLP encoder), features, 10 epochs, 64 windows, even coverage | 128 | 0.23 | 0.51 | 0.26, 0.17 |
| Connectome (MLP encoder), the same and the teacher's re-entry (round 0 shared) | 128 | 0.23 | 0.37 | 0.39, 0.31 |

Reading:
- The control-scale features are what stops the collapse: without them every learner (connectome, MLP, frame-wise MLP) falls to 0 to 0.08 once it drives alone; with them it stays at 0.2 to 0.5.
- With the re-entry the connectome's first beta-0 round is its best (0.39, the training L1 falling from 0.079 to 0.072 as its own states arrive, where every earlier run's rose), above round 1 of runs/skill-dagger-connectome-001 (0.29); its second is 0.31.
  Close_door is solved (1.00 in both), open_drawer and close_drawer reach 0.33 to 0.67, place and stack stay at 0.
- The frame-wise MLP with the features and the re-entry holds 0.44 to 0.54 over four beta-0 rounds; the pipeline's sequence trainer (windows, gradient clipping) stays below it with the same features (the matched MLP 0.17 to 0.33), which a window-versus-frame fit on the same aggregate does not explain (held-out L1 0.146 against 0.140), so part of the gap is run-to-run variance at 12 episodes per skill.
- Remaining stalls: the re-entry connectome is still stationary 42% of its steps after round 3, now mostly (49%) in DESCEND 3.1 mm (quartiles 2.5 to 4.2 mm) from the handle, just outside the teacher's 2 mm gate to close; APPROACH stalls fell from 56% to 25%.
  The next memoryless fix is the same argument one gate further (close from a few millimetres with the finger-contact centring the teacher already has), which may change the teacher table and has to be re-measured; done next (below).

H-G, the teacher's precision gates are tighter than a learner can hold: supported, fixed at the root in the teacher.
- Commands with a floor: while aligning (APPROACH, DESCEND, CARRY) a proportional command smaller than `COMMAND_FLOOR` (0.25 of a full step, 3.5 mm or 12.5 mrad) is raised to it, unless the error is already inside half a floor step (1.75 mm, 6 mrad), so the hand cannot overshoot by more than that.
  A learner a few millimetres off now gets a label that moves it (a plain P command at 3 mm was 0.21, and the regression smoothed such labels to zero), never a near-zero hover.
- Closing gates from geometry instead of 2 mm (handles) and 3 mm (objects): the jaws close once the target lies between the pads with 3 mm to spare along the jaw axis (half the gap the grip settles at, 8 cm fully open, minus the handle's radius or the object's half narrow side), the pads overlap it across the jaw axis (the Panda pad's 8.5 mm half width for a vertical bar or a knob; the lid bar runs across the jaws), and the pads are at the handle's height as before.
  Floored at the old gates, capped at 10 mm for handles and 5 mm for objects, so the coupled fingers still meet an object together; the closing hand keeps servoing to the grasp point and centring on the finger contacts.
  Resulting tolerances: the drawer bar and knobs 10 mm along and 5.5 mm across, the lid bar 3 mm along (pinched from a part-open hand) and 10 mm across, objects 5 mm by 5 mm.
- Teacher table (all 32 cells, 20 episodes each, same seeds): identical successes in every cell (619 of 640 before and after), mean steps to success within 0.5% (docs/results/manipulation-teacher.json).
  The teacher's own path rarely meets the changed rules: it is already under the floors' half steps or inside the old gates when they apply.

Learner-only probe with the teacher's new gates (single subgoals, round 0 the teacher, round 1 beta 0.5, then two rounds with the learner alone; configs as shipped, banks of runs/skill-dagger-connectome-001; mean single-subgoal success over the seven skills, 12 validation-bank episodes per skill; stationary = the hand moved under 5 mm over 10 steps):

| Run | Episodes a round | Round 0 | Round 1 | Learner alone | Stationary share |
| --- | --- | --- | --- | --- | --- |
| Connectome (MLP encoder), before the gate change | 128 | 0.23 | 0.37 | 0.39, 0.31 | 0.42 |
| Connectome (MLP encoder), new gates | 128 | 0.30 | 0.37 | 0.51, 0.55 | 0.30, 0.32 |
| Matched MLP, new gates | 256 | 0.25 | 0.28 | 0.32, 0.52 | 0.37, 0.30 |

The connectome's last round: open_drawer 1.00, close_drawer 1.00, close_door 1.00, open_door 0.50, pick 0.38, place and stack 0; the round is selected on validation and scores 0 of 7 iid episodes from true starts with subgoal fraction 0.18.
The stalls left are mostly APPROACH (53%) and DESCEND (19%); in DESCEND the xy correction asked for is now 1.6 mm at the median (quartiles 1.1 to 4.5 mm), inside the new gates, so these are no longer gate stalls.
Place and stack are still 0 for every learner (see Place and stack, below).

Relaunch (the shipped configs carry the fix: `control_features`, 10 epochs a round capped at 3,000 updates for the connectome and the shuffle, 6,000 for the GRU and 20,000 for the MLP, 64 windows an update; the teacher's re-entry, command floors and geometric closing gates are in the code):

```
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli manipulation skill-dagger --config configs/skill-dagger-connectome.json --output runs/skill-dagger-connectome-002
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli rl manipulation --config configs/ppo-manipulation-skill-dagger.json --output runs/ppo-manipulation-skill-dagger-002
```

The other controllers use configs/skill-dagger-shuffled.json, -mlp.json and -gru.json with their own run directories; a PPO smoke from the fixed connectome run (control features through the frozen encoder, DAPG from its train.npz) completed.
Estimates on the shared GPU (the aggregate of the first run: 0.07, 0.15, 0.23, 0.33, 0.5, 0.69, 0.87, 1.25, 1.63 and 2.0 M steps): connectome or shuffle about 27,700 updates at about 1 s (0.6 s with the GPU free) plus about 1 hour of rollouts and evaluations, 8.5 to 9 hours (5.5 to 6 free); matched MLP about 77,000 updates at 17 ms plus rollouts, about 1.5 hours; GRU about 45,000 updates at about 0.15 s plus rollouts, about 2.5 hours; the PPO from stage 2 as before, 3 to 4 hours.

### Place and stack (branch fix/place-stack)

Every learner scored 0 on single-subgoal place and stack from resets while the other skills reached 0.5 to 1.0; nearly every template has a place or a stack, so this blocks the full run.
Tool: `scripts/place_stack_trace.py` (all 44 place and 12 stack starts of the validation bank, the teacher labelling every state; each episode ends in one reason: `carry`, `turn`, `lower` while still holding, `dropped` (let go away from the target), `touching`, `outside`, `unsettled` after release, split by whether the start holds the object).
One cause at a time, each change re-measured on the full teacher table (acceptance: train cells at 19 or 20 of 20, no cell down by more than 2).

1. Time budget: a single-subgoal reset was cut at 300 steps with no base, where a template gets 200 plus 300 a subgoal.
   A place from a subgoal start is mostly a whole pick and place (36 of 44 starts do not hold the object: they follow an articulation or another placement) plus the 10-step rest test, and the teacher itself needed a median of 297 steps: it solved 50% of the place starts within 300 (open_door 58%, stack 75%).
   Fix: resets get `tk.horizon(budget)`, the templates' own budget (500 for one subgoal); the teacher then solves place 93%, stack and every other skill 100%.
   With it, learners still scored 0: 36 of 44 place episodes of the connectome ended `dropped` (never picked), the teacher in APPROACH 61% of the time.
2. The same path dependence as at the handles, at objects: the learner stood with the hand open at the grasp point (2 mm in xy, 1 mm in height) while a fresh teacher said "rise to the hover", where the teacher's own path, through the hover's alignment gate, says "descend" or "close" (64% of the connectome's stationary steps).
   Fix: the object re-entry (`_object_down`): an open hand already below the hover within 12 mm of the grasp point and turned within 0.15 rad descends or closes, also out of CLEAR, except while the object settles where it was placed (a re-entry there picked it up again: `retrieve` and `unpack` fell to 0 when tried without that exclusion).
3. The pads' height gate: the next stalls were in DESCEND with the hand 8 mm below the grasp point, labelled "rise", a label the teacher's data never has.
   Fix: the pads close anywhere from 8 mm above the grasp point down to where their lower edge would reach the object's bottom, at most 2 cm below (`_pads_on_object`), and within 8 mm (was 5) along the jaws.
4. Contradictory turn labels: in the teacher's own data, 64 to 70% of steps holding the object with a placement heading error over 0.1 rad were labelled yaw 0 (LIFT turned nothing) where CARRY, from the same held states, labels a full turn; learners holding the object stopped turning in CARRY.
   Fix: LIFT turns toward the placement once the object is out of every receptacle.
5. The carry-to-lower gate (8 mm and 0.05 rad): learners holding the object a few millimetres or degrees off stalled in CARRY or `turn`.
   Fix: CARRY also lowers when the object fits where it hangs (`_placement_fits`: every corner inside the receptacle's interior with 4 mm to spare; a stack: its centre over the base's footprint), within 2 cm and 0.25 rad of the target, since LOWER keeps steering; lowering commands get the command floor.

Teacher table after all five (docs/results/manipulation-teacher.json): 625 of 640 against 619 before; four cells up (`shelve` iid 19 to 20, `unpack` unseen objects 16 to 18, `shelve` unseen furniture 16 to 18, `full_cleanup` 17 to 19), `shelve_then_put_away` 20 to 19, every train cell unchanged (139 of 140).
The release, the rest test and containment were not the problem: no learner episode ended `touching` or `outside`, and the teacher's own placements settle.

Learner-only probe (as before: single subgoals, round 0 the teacher, round 1 beta 0.5, then two rounds with the learner alone):

| Run | Place (44 starts) | Stack (12) | Place, starts holding the object (8) | Mean of the seven skills, last two rounds |
| --- | --- | --- | --- | --- |
| Matched MLP, before (fixes 1 and 2 only) | 1/44 | 0/12 | 1/8 | 0.32, 0.52 (skill evaluation) |
| Matched MLP, all five | 10/44 (0.23) | 1/12 | 7/8 | 0.42, 0.55 |
| Connectome (MLP encoder), all five | 10/44 (0.23) | 5/12 (0.42) | 5/8 | 0.53, 0.73 |

The connectome's last round solves open_drawer, close_drawer, close_door and pick from resets (1.00 each, open_door 0.50), and its selected round completes 3 of 7 iid episodes from true starts with subgoal fraction 0.55 (the first successes from true starts of any learner here); it is stationary 34% of its steps.

Remaining failures of place (both learners): 23 and 25 of 44 `dropped` (the pick part: a place start begins after another subgoal, often with the hand at a handle) and 6 to 10 `carry`; the MLP is stationary 41% of its steps.

### Phase cue (branch feat/phase-cue)

docs/ARCHITECTURE_ANALYSIS.md found that one linear map per teacher phase fits the teacher's commands about three times better than one map overall: most of what a controller has to learn is the switching between motor phases.
The memoryless cue therefore gains the current motor phase (`motor_phase`, a one-hot over approach, descend, close, move, lift, carry, lower, release, retreat, clear, before the progress field), computed by `flyarm.manipulation.phases` from scene predicates only, with the teacher's own geometry (alignment at the hover, the closing gates, a pad on the handle at its height, a held object that fits over its destination, the previous subgoal's handle not yet at the teacher's stop) and never the teacher's memory.
`phase_cue: false` (the default) zeroes it, the no-phase-cue control, as `cue: false` zeroes the whole cue.
The observation grows from 220 to 230 features, so checkpoints of earlier runs no longer load.

Agreement with the teacher on its own episodes (`scripts/phase_cue_agreement.py`, three per train template from true starts, 22,074 steps, docs/results/phase-cue-agreement.json): 95.9% with the phase the teacher decides from the same state, 94.0% with the phase it acts in (which lags by one step at every switch).
By skill 87% (close_drawer) to 99% (stack).
Where they differ: the pads already touching a handle while the teacher still counts its 6 settling steps in CLOSE (181 steps, 11% of CLOSE), the teacher's retreat ending on its own counters (reach, not rising) against the observable height (142 and 157 steps between retreat and approach), a pad brushing the handle during the descent (87), an object held but not yet risen (81), and the teacher's retry paths (CLEAR, 58%).

Phase linearity (scripts/manipulation_phase_linearity.py, 42,927 teacher steps, train / held-out episodes, docs/results/manipulation-phase-linearity.json): one linear map 0.246 / 0.581, per skill 0.203 / 0.600, per teacher phase 0.086 / 0.340, per skill and teacher phase 0.063 / 0.274, per observable phase 0.097 / 0.436, per skill and observable phase 0.074 / 0.330.
The observable phase recovers almost all of the teacher phase's gain on the fitted episodes.

Capacity probe with and without the phase cue (scripts/sensory_encoder_probe.py on 28 new teacher episodes, 4 per train template, 28,307 steps, one in seven held out, 12 epochs, matched budgets; docs/results/phase-cue-capacity.json):

| Variant | Held-out L1 without / with the phase cue | Training L1 without / with |
| --- | --- | --- |
| Linear encoder + measured connectome (encoder rate x0.1) | 0.260 / 0.206 | 0.225 / 0.169 |
| MLP encoder + measured connectome | 0.219 / 0.150 | 0.166 / 0.120 |
| MLP encoder + degree-preserving shuffle | 0.196 / 0.153 | 0.158 / 0.125 |
| Matched MLP | 0.165 / 0.107 | 0.110 / 0.078 |

The phase cue lowers every variant's error by 21 to 35%, but the linear-encoder connectome does not reach the MLP: with the cue it fits about as well as the MLP-encoder connectome did without it (0.206 against 0.219), and still twice the MLP's error; the measured wiring and its shuffle remain equal.

Learner-only skill DAgger with the phase cue (as for place and stack: single subgoals, round 0 the teacher, round 1 beta 0.5, then two rounds with the learner alone; against the same runs without the cue):

| Run | Mean skill success, learner-only rounds | Place (44) | Stack (12) | Stationary | True starts, train validation: subgoal fraction (successes of 7) | iid, selected round (7 episodes) |
| --- | --- | --- | --- | --- | --- | --- |
| Connectome (MLP encoder), no phase cue | 0.53, 0.73 | 10/44 | 5/12 | 0.34 | 0.13, 0.41 (0, 2) | 3/7, 0.55 |
| Connectome (MLP encoder), phase cue | 0.76, 0.80 | 20/44 | 12/12 | 0.23 | 0.35, 0.59 (1, 3) | 1/7, 0.35 |
| Matched MLP, no phase cue | 0.42, 0.55 | 10/44 | 1/12 | 0.41 | 0.21, 0.18 (0, 0) | 0/7, 0.25 |
| Matched MLP, phase cue | 0.49, 0.68 | 17/44 | 12/12 | 0.29 | 0.20, 0.40 (1, 1) | 2/7, 0.43 |

The phase cue helps both learners on every single-subgoal measure and on the train-split true starts; the seven iid episodes are too few to separate the connectome runs (1 and 3 successes).
The shipped skill-DAgger configs turn it on; the no-phase-cue control is the same config with `phase_cue: false`.

### PPO for the controls and the final evaluation (branch feat/fair-controls)

The curriculum PPO stage used to accept brain policies only, so the matched MLP and GRU got imitation alone while the connectome got imitation and PPO.
PPO now fine-tunes the controls with the same recipe (reward, curriculum from stage 2, DAPG, critic, clipping, selection on validation) and the comparable trainable part: the last linear map into the action's tanh, everything before it frozen.
- Connectome and shuffle: the linear motor decoder (2,022 readout neurons to 5 actions, 10,115 parameters), encoder and graph frozen.
- Matched MLP: its last linear layer (653 hidden units to 5 actions, 3,270 parameters), both hidden layers frozen.
- Matched GRU: its linear readout (330 hidden units to 5 actions, 1,655 parameters), the recurrent cell frozen; the rollout carries the cell's state and zeroes it at episode ends like the connectome's, and the DAPG features replay each demonstration from a zero state through the frozen cell, as the policy runs.
The trainable part is smaller for the controls than for the connectome (their feature layer is narrower); tuning more of them would give them trainable capacity the connectome does not get.
The config refuses an encoder rate for the controls.
Configs: configs/ppo-manipulation-skill-dagger-mlp.json and -gru.json (runs/skill-dagger-mlp-002 and -gru-002, otherwise identical to the connectome's).
Smoke (16 environments, 3 iterations across the stage switch, DAPG on 2,000 steps, while the queue used the GPU): about 1.5 minutes each, 1,150 to 1,270 steps per second, the base checkpoints scored as expected (GRU 6 of 7 validation episodes, MLP 3 of 7) and the selected checkpoint saved.

`scripts/final_evaluation.py` re-scores any set of checkpoints (imitation or skill-DAgger runs, PPO runs at their selected or any iteration, or any weights file; every policy kind) on the four test splits from true starts, with 32 episodes per template by default from seeds 500 onward in each template's block, disjoint from every episode any checkpoint was selected on (the train split and the first 8 seeds of each held-out block).
Each policy runs with its own run's observation settings; per split it writes the successes, the success rate with a Wilson 95% interval, the subgoal fraction and the motion-quality metrics, into one JSON with a row per label (labels already scored are skipped, so a stopped evaluation continues).
`scripts/manipulation_demo_video.py` renders one successful and one failed episode per template of a split for a checkpoint, side by side and labelled (checkpoint, template, seed, subgoals, current subgoal), from the same seed block.
It runs the single-episode environment, whose episodes match the batched evaluation's but can end differently in contact-rich episodes (the GRU: 14 of 14 iid outcomes agreed, two subgoal counts differed; one unseen-composition success in the batched run failed in the single one), so it finds its own outcomes instead of taking the evaluation's.

```
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli rl manipulation --config configs/ppo-manipulation-skill-dagger-mlp.json --output runs/ppo-manipulation-skill-dagger-mlp-002
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python -m flyarm.cli rl manipulation --config configs/ppo-manipulation-skill-dagger-gru.json --output runs/ppo-manipulation-skill-dagger-gru-002
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/final_evaluation.py --policy connectome=runs/skill-dagger-connectome-002 --policy connectome-ppo=ppo:runs/ppo-manipulation-skill-dagger-002 --policy mlp=runs/skill-dagger-mlp-002 --policy mlp-ppo=ppo:runs/ppo-manipulation-skill-dagger-mlp-002 --policy gru=runs/skill-dagger-gru-002 --policy gru-ppo=ppo:runs/ppo-manipulation-skill-dagger-gru-002 --output docs/results/manipulation-final.json
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/manipulation_demo_video.py --policy ppo:runs/ppo-manipulation-skill-dagger-002 --label "connectome + PPO" --split iid_test --output runs/manipulation/demo-connectome-ppo-iid.mp4
```

### Imitation failure analysis

The first full imitation runs (runs/whole-brain-manipulation-001 and -diagnostic-001 in the main checkout) completed no episode on any split with any controller: subgoal fraction 0.01 to 0.03 for the connectome and the GRU and 0.000 for the MLP, although the MLP fit best (L1 0.078) and the teacher solves 91 to 98% of the same episodes.
Each hypothesis below was tested on its own, in order; the scripts are in `scripts/manipulation_imitation_probe.py` and the session's scratch probes, and every number is from a real closed-loop run.

**H1, a pipeline bug.**
Test 1: the evaluation path (lockstep batched environments, action casting and clipping) driven by an oracle that returns the teacher's action: 36 of 36 episodes succeed (iid test, unseen composition, validation), the teacher's own rate.
Test 2: an MLP overfit to one put_away demonstration (L1 0.021) and replayed on the same seed: the trajectories part at step 2 and the replay completes no subgoal; the first visible divergence is the wrist turn, where the demonstration stops turning at step 30 and the replay turns past it.
Reading: no bug in the plumbing; the single-demonstration replay only shows that a policy off its one trajectory has nothing to go on.

**H2, the teacher is not a function of the observation.**
Test: a large frame-wise MLP (3 x 512) fit on 28 teacher episodes (27,257 steps), L1 by input set, training / held-out episodes:

| Input | L1 | Turning steps with the direction right (held out) |
| --- | --- | --- |
| observation (217) | 0.010 / 0.061 | 70% |
| plus the commanded yaw | 0.010 / 0.072 | 64% |
| plus the teacher's phase | 0.008 / 0.060 | |
| plus all teacher state (integral, anchor, centring, counters) | 0.007 / 0.056 | |

The aggregate is nearly Markov, but one decision is not learnable from the observation: the direction of the wrist turn.
The teacher turns by the half-turn-wrapped difference between the heading it needs and the commanded heading, flipped near the yaw limit, and the commanded heading is an integrator inside the action that the observation did not contain (only the lagging actual jaw angle); half the turns go each way.
Fix: the heading rules move from the teacher into the sim (`pick_headings`, `articulation_headings`, `place_headings`, `turn_to`; the teacher calls them and its actions are bit-identical over 36 episodes x 1,500 steps), the cue carries the turn to the grasp and to the place heading, and the commanded yaw joins the proprioception (observation 217 to 220).
With it the rule clip(turn / 0.05) matches the teacher's turn on 100% of 2,846 turning steps, and the frame MLP's held-out dyaw error falls from 0.069 to 0.028.
A second hidden state came out of H4 (below): the teacher's xy integral.

**H3, averaged multimodal actions (the gripper).**
Test: the trained diagnostic MLP on the teacher's own validation states.
The gripper is not the problem: its sign is right on 98.4% of the steps where the teacher commands it; the wrist turn is: right on only 34% of turning steps, L1 0.66 there, because the turn direction is split 50/50 over the data (H2).
No discretized gripper head was needed.

**H4, compounding error; and what the closed-loop traces showed.**
Tracing the trained MLP against the teacher's labels, episode by episode, gave three findings in turn:

1. The copycat problem.
   A frame MLP fit to L1 0.012 on 56 episodes outputs a zero action from the start pose for 100 steps while the teacher says (-1, +1, 0, +1, +1).
   In the demonstrations the arm rests only for the four steps in which the teacher opens the hand, and is moving afterwards, so "do what the joint velocities say" fits the data.
   With joint and object velocities zeroed in its input the same MLP leaves the start correctly.
   The kitchen protocol excludes velocities for the same reason (research log E4).
   Fix: an environment switch `velocities` (the critic's privileged observation keeps them); `velocities: false` is the imitation default.
2. The arm sank under its own weight.
   A zero action moved the hand 88 mm down in 50 steps (1.8 mm per step): actions are Cartesian steps from the measured joint positions, every position servo holds its target short by its gravity load, and the next step starts from the sagged pose.
   The teacher fought it with an integral term on its xy error, hidden state that sat at its 12 mm clip on 22% of the steps where it was active.
   Fix: the arm is gravity-compensated, as a real Panda is (drift 0.1 mm in 50 steps at home, 2 mm at a low far pose), and the integral is removed.
   Quick teacher checks (8 episodes per training template): 84% with gravity compensation alone (the limit cycle below), 96% with the armature fix, 98% with the integral removed as well.
   Two teacher rules had silently relied on the sag and were made explicit: while travelling the hand sinks toward the transit height at 2 mm per step (the sag's rate, which keeps the time budget), and the lid is opened with the handle led 0.16 rad along its arc instead of 0.08 (lifting the lid's weight takes that much servo error; closing keeps 0.08, since a longer lead pulled the pinch off the bar).
   Gravity compensation exposed a numerical limit cycle: near an upright shoulder joints 1 and 3 are coaxial, their counter-rotation mode has almost no inertia, and the two actuators swapped between +87 and -87 Nm every 2 ms step with the hand still; arm armature 0.3 (Menagerie 0.1) removes it.
3. Underfitting by the trainer.
   The sequence trainer walks batches of whole episodes window by window, so consecutive Adam updates see nearly the same states; the pipeline's MLP ended at L1 0.096 after 80 epochs (0.194 after 12) where frame-wise training reached 0.016.
   Fix: window sampling (`sampling: windows`): every update fits 32 windows of 16 steps drawn uniformly from all demonstrated steps, each entered after a 16-step burn-in without gradient from the zero state (the connectome keeps information for about 0.2 s at the default gain, E15 and E16); the pipeline's MLP reaches L1 0.053 in 40 epochs.

With the three fixes the remaining gap is the ordinary one, compounding error, which DAgger closes step by step.
A frame-trained MLP with three DAgger rounds: subgoal fraction 0.000, 0.056, 0.093, 0.139 on 21 held-out train-split episodes, with the first subgoal (opening a drawer) done in 11 of 12 drawer episodes by round 3.

**Results through the pipeline** (`scripts/manipulation_imitation_probe.py`, 21 test episodes of the train split that no stage trains or selects on, three per template; at least k subgoals per template):

| Run | Training L1 (train / val) | Success | Subgoal fraction | Episodes with at least 1, 2, ... subgoals |
| --- | --- | --- | --- | --- |
| MLP with the heading features, before the physics and velocity fixes (episode sweep, behavior cloning only), 8 demos per template | 0.084 / 0.106 | 0/21 | 0.000 | none |
| the same with the heading features zeroed | 0.083 / 0.097 | 0/21 | 0.000 | none |
| MLP, all fixes, episode sweep, 12 epochs, 1 DAgger round | 0.194 / 0.200 | 0/21 | 0.000 | none |
| MLP, all fixes, episode sweep, 80 epochs, 1 DAgger round | 0.117 / 0.136 | 0/21 | 0.000 | none |
| MLP, all fixes, window sampling, 40 epochs, 3 DAgger rounds (8 per template) | 0.081 / 0.104 | 1/21 | 0.143 | put_away 3, 0, 0; retrieve 2, 1, 1, 1; tidy 2; unpack 2, 1 |
| Connectome, all fixes, window sampling, 4 demos per template, 15 epochs, 2 DAgger rounds | 0.241 / 0.242 | 0/21 | 0.012 | shelve 1 |
| MLP, as the 3-round run, on the final teacher (travel sink and lid leads added after the runs above) | 0.085 / 0.109 | 0/21 | 0.075 | put_away 2; retrieve 3; tidy 1 |

The MLP now completes whole episodes (a retrieve: open the drawer, pick, place in the bin, close) and its first skill, opening a drawer, succeeds in 9 of 12 drawer episodes (6 of 12 in the repeat on the final teacher; with 21 test episodes and one seed these probes differ by a few episodes); DAgger rounds keep adding (best validation subgoal fraction 0.01, 0.03, 0.05, 0.11 for behavior cloning and rounds 1 to 3), so the full configs now run four rounds.

**The connectome is a near-linear controller here.**
Its fit stalls at L1 0.22 to 0.24 where the MLP reaches 0.05 to 0.08, so the question is what the frozen connectome can express through the B1a interface.
Test: frame-wise fits on the same 28 demonstrations (one episode in seven held out), a linear readout of the connectome's features against policies on the raw observation:

| Model | L1 train / held out |
| --- | --- |
| linear policy on the observation, tanh(W x + b) | 0.211 / 0.267 |
| decoder only on connectome features, untrained calibrated encoder | 0.216 / 0.247 |
| decoder only on connectome features, the probe's trained encoder | 0.195 / 0.237 |
| MLP 3 x 512 on the observation | 0.021 / 0.140 |

The same decoder-only fit across the rate model's regimes stays at the linear level or worse: recurrent gain 0.99 (0.214 / 0.245), four times the input drive (0.202 / 0.280), weight-norm power 2 (0.262 / 0.308), gain 0.99 with power 2 and four times the drive (0.234 / 0.282).
Reading: with a linear encoder and a linear readout the frozen connectome adds no usable nonlinearity for this task, as research log E27 found on the kitchen (the fly controller fit the demonstrations about as well as a linear policy); the teacher's decisions (descend once aligned, close once centred, switch heading when the object is held) are switches a linear policy cannot make.
The pipeline is no longer the limit for the connectome; this is a property of the controller class, and what to do about it (reward fine-tuning from the weak imitation start, as on the kitchen, or a richer interface) is a research decision, not a fix.

**What changed** (all verified by tests and the runs above): the arm is gravity-compensated with armature 0.3 (docs of `scene.py`), the teacher's integral is removed and its heading rules live in the sim, the observation gains the commanded yaw and two heading errors (220 features), `velocities: false` is the imitation default, window sampling is the imitation default, and the imitation configs run four DAgger rounds.
The teacher table below was re-measured on the changed task.


### Costs and wall-clock estimates

Measured with `scripts/manipulation_training_throughput.py` (docs/results/manipulation-training-throughput.json), Apple M5 Max, 4 simulation threads, other jobs sharing the machine:

| What | Measured |
| --- | --- |
| Environment alone, random actions, 128 environments | 2,310 steps/s (1,590 before the contact exclusion) |
| Connectome closed loop (encoder, MaleCNS, decoder), 128 environments, fresh episodes | 1,260 steps/s (1,310 before; 22 to 26 ms of each control step is the connectome, and GPU sharing moves this row by about 5%) |
| Connectome closed loop, 32 environments | 1,200 steps/s (1,140 before) |
| Behavior cloning with the encoder training, episode sweep, batch 16, BPTT 16 | 3.0 to 3.7 ms per demonstration step (two measurements, 21 episodes; no physics involved) |
| Behavior cloning with the encoder training, window sampling (32 windows of 16 steps, 16-step burn-in), connectome probe | 1.2 to 1.6 ms per demonstration step (28,842 steps in about 46 s per epoch; 69,000 in 81 s) |
| Teacher demonstrations, one environment | 21 episodes (21,900 steps) in 37 s (41 before) |
| Evaluation of a failing policy, one episode per template of all four held-out splits (25 episodes) | 64 ms per lockstep step, 2.8 minutes (140 ms and 6 minutes before) |

PPO at 128 environments and 64-step rollouts (the smoke run below, after the contact exclusion): 820 steps/s in the first iteration and 550 to 580 in the next five, about 14.5 s per iteration including the update (555, then 410 to 450, and 19 s before the exclusion).
The environment still sets the pace once a policy drives into contact: over steps 300 to 400 of an undertrained policy, one step of 128 environments took 176 ms of physics and task rules, 24 ms of privileged observation and 22 ms of connectome (264, 20 and 22 ms before the exclusion).

Estimates for the shipped configs, after the contact exclusion (a policy that fails runs every evaluation episode to its horizon, so evaluations are upper bounds; a DAgger round's failing rollouts also run to their horizons, which is what makes the aggregate grow):

| Stage | configs/whole-brain-manipulation.json | configs/ppo-manipulation.json |
| --- | --- | --- |
| Data | demonstrations and validation episodes 3 to 6 min, teacher evaluation 3 to 5 min | DAPG replay about 2 min |
| Training | behavior cloning 12 epochs x 4 to 5 min (175,000 steps at 1.2 to 1.6 ms, window sampling); 4 DAgger rounds of 56 learner episodes (about 100,000 steps each, 3 min of rollouts), then 6 epochs on aggregates of about 275,000, 375,000, 475,000 and 575,000 steps: about 4 hours together | 1,000 iterations x 10 to 15 s (8.2 M steps at 820 to 550 steps/s) |
| Selection and evaluation | about 11 closed-loop validations x 1 to 1.5 min, final evaluation 7 to 11 min | 11 evaluations x 3 to 6 min |
| Total | about 5 to 6 hours for configs/whole-brain-manipulation.json after the imitation fixes (four DAgger rounds; the aggregate dominates) | about 3.5 to 5 hours |

### Smoke runs

Tiny imitation (one demonstration, validation and test episode per template, 2 behavior-cloning epochs, one DAgger round of one episode per template), `flyarm manipulation imitate`, 19 minutes end to end: the teacher succeeded in 7 of 7 demonstrations (7,154 steps); the training L1 went 0.43 (decoder only) to 0.36 when the encoder joined (validation 0.49 to 0.38); the DAgger rollout (beta 0.5) completed 1 of 7 episodes and 27% of subgoals; validation success was 0 of 7 for both phases, so behavior cloning was kept; the teacher solved 24 of 25 test episodes and the controller none, as expected at this size.
Before the encoder's learning rate was separated, the same smoke's L1 rose from 0.43 to 0.64 at that point.

PPO from that checkpoint, 128 environments, 6 iterations, critic warm-up 2, DAPG on 5,000 demonstration steps, `flyarm rl manipulation`, 12 minutes (repeated after the contact exclusion for the rates above): the demonstration loss fell from 1.24 to 0.78, the critic's loss from 0.14 to 0.07, and the base and final evaluations (one test episode per template of all four held-out splits plus one validation episode per template, all in lockstep) ran and selected on validation.
`flyarm rl watch --run runs/smoke-manip-ppo --once` rendered its two-episode progress clip in 45 s.

### Open items

- Lesion evaluations of the connectome checkpoint (edges off, direct synapses only, state reset every step) are not wired in yet; the kitchen and B1a pipelines have them.
- PPO starts from brain checkpoints only; the reward-only MLP control of the kitchen (E46) is not ported.
- The DAPG term uses the teacher's demonstrations only, not the DAgger labels.
- The no-cue control is `cue: false` in the imitation config; a PPO run inherits the imitation run's setting.
