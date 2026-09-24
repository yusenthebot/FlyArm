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

217 floats (`OBS_DIM`), in fixed order (`flyarm.manipulation.sim.*_FIELDS`):

- robot, 20: joint positions and velocities, end-effector position, gripper yaw (cos, sin), gripper opening;
- four object slots, 22 each: presence mask, position, position minus end effector, 6-D rotation, the grasp task's 5-D shape descriptor, linear velocity, grasped flag (zero when the slot is empty);
- furniture, 46: drawer cabinet xy and facing, drawers present, open fractions, both handle positions and their offsets from the end effector, handle type, cabinet xy and facing, lid angle, lid handle position and offset, lid handle type, shelf height, cabinet interior size, bin xy, facing and size, region xy and size;
- the cue, 63: the current subgoal's skill one-hot (7), which articulation (3), object slot one-hot (4), the object's position, offset from the end effector, rotation and descriptor, the receptacle one-hot (5 receptacles or "on an object"), the target point for the object's bottom and its offsets from the end effector and from the object, the receptacle's half size and facing, the handle position and its offset, the joint's open fraction, the direction the handle must move, and progress (done over total, left over 8).

The cue is memoryless: it is recomputed from the scene each step through the same predicates as the reward, so a policy never needs to remember where in the task it is.
`cue=False` zeroes the whole cue block (the no-cue control); the rest of the observation is unchanged.
The privileged observation (230, for a critic) adds object masses, the per-subgoal done flags and the commanded yaw.

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
- pick, place and stack: the grasp task's rule (jaws across the narrow side, or along a drawer's width when picking from one and that is still across the object, so no finger lands on the front panel; pads at the geometry-derived height, integral xy alignment, closes only when centred within 3 mm), carry above every furniture top and object, turn to the receptacle's axis (finishing the turn before coming within 15 cm of the target, so the hand does not sweep an opened lid shut), lower until the bottom reaches the target surface or the descent stalls, release and rise.

Its memory is the subgoal it is executing, a phase, a counter and a held xy; watchdogs restart a subgoal whose phase stalls.
It re-derives its phase from the scene whenever the current subgoal changes, so it can label any state a learner reaches (DAgger), and it never writes simulator state.

### Teacher table

Successes over 20 episodes per cell, all cells of a split run in one batched environment on that split's own seed block (`docs/results/manipulation-teacher.json`, with mean steps to success and the subgoal each failure was stuck at).

| Task | train | iid test | unseen objects | unseen furniture | unseen composition |
| --- | --- | --- | --- | --- | --- |
| `put_away` | 20/20 | 20/20 | 20/20 | 20/20 |  |
| `retrieve` | 20/20 | 20/20 | 19/20 | 20/20 |  |
| `shelve` | 18/20 | 19/20 | 18/20 | 15/20 |  |
| `tower` | 20/20 | 20/20 | 20/20 | 20/20 |  |
| `sort` | 20/20 | 20/20 | 14/20 | 19/20 |  |
| `tidy` | 20/20 | 18/20 | 19/20 | 18/20 |  |
| `unpack` | 20/20 | 18/20 | 18/20 | 18/20 |  |
| `shelve_then_put_away` |  |  |  |  | 17/20 |
| `retrieve_to_shelf` |  |  |  |  | 15/20 |
| `tower_then_put_away` |  |  |  |  | 20/20 |
| `full_cleanup` |  |  |  |  | 17/20 |

Every train cell is at 90% or above (138/140 overall); every iid cell too.
Cells under 90%, all on held-out splits:

- `sort` on unseen objects (14/20): all six fail at the stack. `stable_on` is a rule on bounding boxes and masses tuned on training objects; on training objects the stack failures it removed were tops tipping off their base on release, and these six were not diagnosed further.
- `shelve` on unseen furniture (15/20): the held-out cabinet (knob on the lid, turned further from the robot, lower shelf) makes every subgoal slower, and in four of the five failures both objects are on the shelf and the horizon ends while the teacher is still closing the lid.
- `retrieve_to_shelf` (15/20): the only template that works a drawer while the lid stands open; failures are picks from the drawer (a finger on the front panel pushing the drawer back under its threshold, reduced but not removed by picking with the jaws along the drawer's width) and the hand knocking the open lid while stepping back from it.
- `shelve_then_put_away` and `full_cleanup` (17/20 each): each failure is stuck at a placement; not diagnosed further.

The teacher's episodes are what imitation would learn from, so a teacher failure is also a state the learner will not see solved; DAgger labels still come from the same controllers.

## Throughput

128 batched environments on the train split, 4 simulation threads (`FLYARM_SIM_THREADS=4`), Apple M5 Max: 1,350 environment steps per second with random actions and 1,130 with the teacher in the loop in the recorded run (`docs/results/manipulation-teacher.json`, `throughput`); an otherwise identical run with less load on the machine gave 1,870 and 1,390.
Physics is 75 to 85% of a step (25 MuJoCo steps over 43 furniture geoms, the no-slip solver and up to four object meshes); the task rules, observation and reward cost 10 to 16 ms per step for all 128 environments, 5 ms of it the hull containment test.
That is five times slower than the grasp task per environment step, and the horizons are 5 to 13 times longer.

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

The controller is the B1a whole-body interface: the 217 observation features go through a trainable linear encoder into the 1,846 ascending neurons of the frozen MaleCNS (166,700 neurons), the 1,314 descending and 708 VNC motor neurons are read out through a frozen unit-norm calibration (research log E26) and a trainable linear decoder gives the 5 actions, one per control step.
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
3. Behavior cloning: L1 loss, per-step weights that give every skill the same total weight (capped at 5 times the mean), batches of episodes of similar length (a batch runs until its longest episode ends), 16-step BPTT windows with the state carried across windows, 2 decoder-only warm-up epochs.
   Episodes stay in host memory and each batch is copied to the GPU cut at its longest episode.
4. DAgger rounds: the learner drives `dagger_episodes_per_template` new episodes per template, the teacher labels every visited state (it re-derives its phase from the scene), the teacher's action is executed instead with probability 0.5 in round 1 and 0.25 in round 2, and training continues on the aggregate.
5. Closed-loop selection on validation episodes of the train split: within behavior cloning every `select_every` epochs, and among the phases at the end (research log E33); success rate first, the mean fraction of subgoals done breaks ties.
6. Evaluation of the selected checkpoint and of the teacher on the same test episodes of every held-out split: success, subgoals completed, mean steps to success, per template, and the motion-quality metrics (stray-contact fraction, mean |a|, saturated fraction, mean |change of a|, final disturbance).

The encoder trains at a tenth of the decoder's learning rate (`encoder_learning_rate_scale`).
Measured on 21 demonstration episodes, three epochs from the same calibrated start (L1 by quarter of epoch 3; input-current RMS into the ascending neurons after epoch 1):

| Encoder | Epoch 3 L1 | Input RMS |
| --- | --- | --- |
| frozen (decoder only) | 0.26 to 0.27 | 0.48 |
| same rate as the decoder | 0.25 to 0.30 | 2.15 |
| a tenth of it | 0.23 to 0.26 | 0.51 |

At the full rate, Adam's early steps move all 217 x 1,846 encoder weights by about the learning rate on correlated inputs and drive the ascending neurons into tanh saturation within one epoch, the input-side counterpart of the readout effect in E26; the kitchen's whole-body interface had 30 inputs and did not show it.
The controls (`policies`: `shuffled`, `gru`, `mlp`) run through the same code; their input modules train at the full rate.

Seeds (flyarm.manipulation.rollout): episode i of a split's template t has seed `seed_start + offset + 1000 t + i`; demonstrations use train offset 0, validation 100,000, DAgger round r 200,000 + 10,000 r, PPO training episodes start at 1,000,000 (below 3,000,000, where the first held-out block starts), and test episodes use each held-out split's own block.
Adding episodes never changes existing ones.

### PPO (flyarm.rl.ppo_manipulation)

`ManipulationTask` is the `TaskAdapter` for the unchanged `train_ppo`; the trainer gained two optional hooks (per-episode horizons for the critic's time feature, and a selection key).

- Reward: the environment's RewardConfig built from the config: subgoal bonus 50 paid once and in task order, potential shaping at weight 10, stray contact 0.05, disturbance 5, action 0.01, smoothness 0 (the kitchen used it; it is off by default here until a run shows commands need it).
  Every level term is a penalty, so the stalling floor is 0 at any gamma and the config's check recomputes it for its own gamma (default 0.995: the next subgoal, about 300 steps away, is discounted to 0.22 instead of 0.05 at 0.99).
- Critic: the 230-feature privileged observation plus the step over the episode's own horizon.
- DAPG: the squared error to the imitation run's own teacher demonstrations (train.npz), at most `bc_max_steps` steps drawn with the skill-balanced weights, replayed once through the frozen encoder and connectome from a zero state; it needs the encoder frozen (`encoder_lr` 0).
- Warm start from the imitation checkpoint; `init_checkpoint` continues from a PPO checkpoint (the critic starts fresh, so `critic_warmup` stays above 0; 50 by default, E39, E46).
- Advantage clip 10 (E39), exploration log std -1.2.
- Evaluation every `eval_every` iterations: `eval_episodes_per_template` test episodes of every split in `eval_splits` and `val_episodes_per_template` validation episodes of the train split, all in lockstep; checkpoints are selected on validation only.
- Training curves carry `subgoals_per_episode` (mean over finished episodes) and the batch peak `max_subgoals_in_one_episode`.
- `flyarm rl watch` re-records one clip of the newest checkpoint (two train-split episodes from seed 60,000, templates by seed) into `progress/latest.mp4`, as for the kitchen.

### Costs and wall-clock estimates

Measured with `scripts/manipulation_training_throughput.py` (docs/results/manipulation-training-throughput.json), Apple M5 Max, 4 simulation threads, other jobs sharing the machine:

| What | Measured |
| --- | --- |
| Environment alone, random actions, 128 environments | 1,590 steps/s |
| Connectome closed loop (encoder, MaleCNS, decoder), 128 environments | 1,310 steps/s (22 ms of each control step is the connectome) |
| Connectome closed loop, 32 environments | 1,140 steps/s |
| Behavior cloning with the encoder training, batch 16, BPTT 16 | 3.0 ms per demonstration step (21 episodes; 2 to 3 ms expected on the full set, whose batches are full) |
| Teacher demonstrations, one environment | 21 episodes (21,900 steps) in 41 s |
| Evaluation of a failing policy, one episode per template of all four held-out splits (25 episodes) | 140 ms per lockstep step, 6 minutes |

PPO at 128 environments and 64-step rollouts (the smoke run below): 555 steps/s in the first iteration and 410 to 450 afterwards, about 19 s per iteration including the update.
The environment, not the connectome, sets the pace once a policy drives into contact: profiled over steps 300 to 400 of an undertrained policy, one step of 128 environments took 264 ms of physics, 20 ms of privileged observation and 22 ms of connectome (73, 10 and 23 ms right after reset).
Two contact sources dominate (single-environment trace): a closed lid resting on the cabinet walls always makes 20 contacts, because the cabinet is welded to the world and MuJoCo's parent-child filter does not apply, although the hinge's joint limit already holds the lid at 0; and a policy that closes the hand on nothing makes up to 44 finger-to-finger contacts and presses the fingers into the table.
Excluding the lid-to-cabinet and finger-to-finger pairs would remove most of them, but it changes the benchmark's physics after its teacher table was published, so it is left as a decision.

Estimates for the shipped configs (a policy that fails runs every evaluation episode to its horizon, so evaluations are upper bounds):

| Stage | configs/whole-brain-manipulation.json | configs/ppo-manipulation.json |
| --- | --- | --- |
| Data | demonstrations and validation episodes 5 to 10 min, teacher evaluation 5 to 8 min | DAPG replay about 2 min |
| Training | behavior cloning 12 epochs x 6 to 9 min; DAgger 2 rounds x (2 min of rollouts + 4 epochs x 9 to 18 min) | 1,000 iterations x 8 to 19 s (8.2 M steps at 1,200 to 430 steps/s) |
| Selection and evaluation | about 6 closed-loop validations x 1.5 min, final evaluation 8 min | 11 evaluations x 5 to 8 min |
| Total | about 3 to 5 hours | about 3.5 to 7 hours |

### Smoke runs

Tiny imitation (one demonstration, validation and test episode per template, 2 behavior-cloning epochs, one DAgger round of one episode per template), `flyarm manipulation imitate`, 19 minutes end to end: the teacher succeeded in 7 of 7 demonstrations (7,154 steps); the training L1 went 0.43 (decoder only) to 0.36 when the encoder joined (validation 0.49 to 0.38); the DAgger rollout (beta 0.5) completed 1 of 7 episodes and 27% of subgoals; validation success was 0 of 7 for both phases, so behavior cloning was kept; the teacher solved 24 of 25 test episodes and the controller none, as expected at this size.
Before the encoder's learning rate was separated, the same smoke's L1 rose from 0.43 to 0.64 at that point.

PPO from that checkpoint, 128 environments, 6 iterations, critic warm-up 2, DAPG on 5,000 demonstration steps, `flyarm rl manipulation`, 12 minutes: the demonstration loss fell from 1.24 to 0.78, the critic's loss from 0.14 to 0.07, and the base and final evaluations (one test episode per template of all four held-out splits plus one validation episode per template, all in lockstep) ran and selected on validation.
`flyarm rl watch --run runs/smoke-manip-ppo --once` rendered its two-episode progress clip in 45 s.

### Open items

- Lesion evaluations of the connectome checkpoint (edges off, direct synapses only, state reset every step) are not wired in yet; the kitchen and B1a pipelines have them.
- PPO starts from brain checkpoints only; the reward-only MLP control of the kitchen (E46) is not ported.
- The DAPG term uses the teacher's demonstrations only, not the DAgger labels.
- The no-cue control is `cue: false` in the imitation config; a PPO run inherits the imitation run's setting.
