# Generalizable grasping environment

The Panda grasps one everyday object, lifts it and holds it.
The object is one of 41 real scanned household objects in eight shape families.
Controllers train on 27 of them and are tested on 14 held-out objects they never saw.
The code is `src/flyarm/grasp/`; nothing in it is wired into the imitation or PPO pipelines yet.

## Task

Each episode puts one object from the configured split on the table and starts the Panda at the pick-and-place home posture (with the same +/- 0.012 rad joint jitter).
The object's position is uniform in x in [0.33, 0.58] m and y in [-0.18, 0.18] m (the robot base is at the origin, the home end effector at x = 0.41), and its yaw is uniform in [-pi, pi).
The horizon is 200 control steps (10 s at 20 Hz).

A reset with seed `s` draws, in order, the seven joint jitters, the table position, the yaw and the object index from `numpy.random.default_rng(s)`.
The pose therefore depends on the seed alone, not on which objects are in the model, and `reset(seeds=..., objects=...)` can fix the object without changing the pose.

## Success rule

An episode succeeds when the object's origin (the centre of its oriented bounding box) is at least 10 cm above its resting height and both fingers touch it, for 20 consecutive control steps.
Finger contact is read from two MuJoCo contact sensors per object (object geom against each finger body), as in `batched_pick_place`.
The count restarts whenever either finger loses contact or the object drops below the lift target.
Success terminates the episode.

## Action

`[dx, dy, dz, dyaw, gripper]`, each in [-1, 1].
It is the pick-and-place control scheme extended with yaw: xyz is a 1.4 cm Cartesian step of the end-effector site, `dyaw` turns the commanded gripper yaw by up to 0.05 rad per step about the world vertical (clipped to +/- 1.75 rad from the reset orientation, more than a half turn of the jaws), and `gripper` maps to the Menagerie actuator exactly as in pick-and-place (-1 closes, +1 opens).
Each action runs 25 MuJoCo steps of 2 ms.
The damped least-squares IK is the pick-and-place one (0.25 rotation weight, 0.055 rad joint clip) plus a null-space pull toward the home posture; without it the redundant elbow drifted into joint limits near the base once the gripper turned, and most objects failed at the same near-base pose.

## Observation

The controller sees 54 numbers (`flyarm.grasp.task.OBS_FIELDS`):

| Field | Size | Meaning |
|---|---|---|
| joint_pos, joint_vel | 7 + 7 | arm joints |
| ee_pos | 3 | end-effector site |
| gripper_yaw | 2 | sin 2phi, cos 2phi of the jaw-closing axis (parallel jaws are symmetric under a half turn) |
| gripper_opening | 1 | 0 closed, 1 open |
| object_pos, object_minus_ee | 3 + 3 | object origin |
| object_rot6d | 6 | first two columns of the object rotation matrix (catches tipping) |
| object_axis_yaw | 2 | sin 2psi, cos 2psi of the object's narrow axis |
| relative_yaw | 2 | sin 2(psi - phi), cos 2(psi - phi) |
| object_vel | 3 | linear velocity |
| object_descriptor | 5 | box long, narrow, height; grasp offset along the long axis; pad height |
| grasp_minus_ee | 3 | geometry-derived grasp target for the site, minus the site |
| lift_target, lift_error | 1 + 1 | object height that counts as lifted, and the remaining rise |
| contacts | 2 | left and right finger on the object |
| ever_grasped, lifted, hold_fraction | 1 + 1 + 1 | episode flags and the hold counter over 20 |

The descriptor is what should let a policy generalize to shapes it never saw.
Every entry is computed from the object's convex hull by a fixed rule (next section), not tuned per object, so a perception system that sees the object could produce it.
`observation(privileged=True)` appends object mass, body-frame angular velocity and the commanded yaw for a critic (59 numbers).

## Reward

`flyarm.grasp.task.shaped_reward`, in the pick-and-place style:

- reach: 0.5 (1 - tanh(10 d)), with d the distance from the site to the grasp target;
- align: 0.25 times reach times cos^2 of the jaw-to-narrow-axis angle;
- grasp: 0.5 when both fingers touch;
- lift: 1.0 times grasped times the fraction of the 10 cm rise achieved;
- hold: 1.0 when grasped and lifted;
- action cost: -0.01 |a|^2 (at most 0.05);
- success: a terminal bonus of 400.

The invariant: the per-step terms lie in [0, 3.25] before the action cost, which only subtracts, and the terminal bonus must exceed the stalling floor 3.25 / (1 - gamma), 325 at gamma = 0.99.
Otherwise a policy that hovers in the best non-terminal state (grasped and lifted but never held for 20 steps) earns as much, discounted forever, as one that finishes.
`RewardConfig` raises if the bonus does not exceed the floor for its `gamma`; pass a larger bonus with a larger gamma.
The reward shapes training only; evaluation reports the success rule.

## Objects

All objects come from Google Scanned Objects in the MJCF conversion of `kevinzakka/mujoco_scanned_objects`, fetched at upstream commit `6ff8d275cebfd5b47e49685e3cfbe64b20e49a3c`.
Meshes and textures are CC BY 4.0; the MJCF conversion is MIT.
`src/flyarm/grasp/catalog/objects.json` records, per object, the upstream name, source URL, licence, SHA-256 of the mesh and the texture, scale, body frame, oriented bounding box, mass and grasp point.
The files are not committed (`assets/` is git-ignored); `scripts/fetch_grasp_objects.py` downloads them and refuses any file whose digest differs.

Geometry is measured from each scan's convex hull (which is also its collision shape) by `measure_mesh`:

- z stays up, as scanned; x and y are the long and narrow sides of the minimum-area rectangle around the hull's footprint, so the jaws close across y;
- the scale is the largest (at most 1, rounded down to 0.01) that makes the narrow side at most 5.5 cm (the open pads are 6.8 cm apart), the long side at most 16 cm and the height at most 10 cm (the fingers are about 5 cm long, so a taller object could only be taken by its top);
- mass is 350 kg/m^3 times the hull volume, clipped to [30, 300] g;
- the grasp point is the hull's centre of mass projected on the long axis;
- the pad height is the lowest height at or above the centre of mass where the hull's cross-section there is within 1 mm of its widest, inside the band the palm allows, and never below the finger floor (1.3 cm).

The last rule matters: pads on a wall that narrows upward (a bottle's shoulder) wedge the object down and out, and pads below the centre of mass leave it balancing on the jaw axis.
Objects that cannot fit the gripper at a height of at least 2 cm are rejected at measurement (the screwdriver, whose 2 mm shaft carries the grasp point, is the one rejection of the final candidate list).

Two contact settings are specific to this scene and documented in `scene.py`.
Object contacts use a 0.004 s time constant (twice the timestep): MuJoCo's contacts act at the acceleration level, so with the default 0.02 s a 30 g mug squeezed by the Menagerie gripper sank 6 to 10 mm into the fingers, the contact normals tilted and the squeeze pushed it sideways out of the jaws, and mugs and bowls failed in 12 to 50 percent of the episodes of an 8-episode probe.
Object contacts also use `condim` 4, so the rubber pads' torsional friction resists the object turning about the jaw axis; on its own this did not change teacher success, the stiffer contacts did.
The Menagerie gripper actuator and the FlyArm pads are unchanged.

## Split

`scripts/grasp_split.py` drops every object whose teacher success over 20 episodes is below 0.8, then splits the rest per family with a seeded shuffle (test fraction 1/3, seed 0), so both splits contain every family.
The split is committed as `src/flyarm/grasp/catalog/split.json`.

| Family | Train | Held-out test | Dropped |
|---|---|---|---|
| mug | bead_mug, classic_blue_mug, sand_cup | paper_cup, white_yellow_mug | |
| bowl | glazed_ramekin, scirocco_bowl, turquoise_bowl | cereal_bowl | |
| bottle | coq10_bottle, htp_bottle, sprinkles_jar | supplement_tub, tonic_bottle | |
| can | cocoa_canister, instant_coffee_jar, xylitol_tub | camera_lens, coffee_can | |
| box | caplet_box, hair_color_box, lipstick_box, raisinets_box | cereal_box, crayon_box | |
| tool | hammer, hand_bell, tape_measure | flashlight, tape_roll | can_opener (0.05), lime_squeezer (0.45) |
| toy | android_figure, henry_engine, lion_figure, mario_figure, yoshi_figure | school_bus, triceratops | |
| rounded | fruit_basket_toy, ladybug_bead, whale_whistle | rubber_chew_toy | pineapple_maraca (0.65), bird_rattle (0.00) |

27 train, 14 held out, 4 dropped.
The "rounded" family holds the fruit-like and organic toys (a fruit basket, a ladybug, a chew toy, a whale); the Scanned Objects collection has no real fruit.
The bird rattle falls over at rest (its scanned pose is not stable, it settles 74 degrees from upright) and the pineapple maraca settles 26 degrees tilted.
The can opener and the lime squeezer are grasped in every episode but mostly slip out while lifted or held (lift rates 0.85 and 0.80, success 0.05 and 0.45); the cause was not investigated further.
Sizes, masses and per-object teacher success are in `docs/results/grasp-teacher.json` and the catalog.

## Scene

mjbatch copies one model for all of its simulations, so a batch that mixes objects needs every candidate object in that one model.
The model holds every object of the chosen split, each on its own free joint.
The episode's object is placed on the table and every other object floats motionless 2.6 m behind the video camera, each at its own spot, held up by an external force equal to its weight (`xfrc_applied`, which is part of the per-simulation state).
A test checks that parked objects do not move and that one batch holds a different object in every environment.

## Scripted teacher

`GraspTeacher` (privileged, `flyarm.grasp.teacher`) recomputes everything from the object's current pose every step: approach above the grasp point with the jaws turned across the narrow axis (within the yaw limit), descend to the geometry-derived pad height, close and wait for six steps of two-finger contact, then rise until the object itself is 3 cm past the lift target, and hold.
Its only memory is a phase and three counters; missed closes, drops and a moved object send it back to the approach.
A resync rule re-derives the phase from physical state that its own rollouts never visit (already squeezing while approaching, centred and open below the hover height), so it can label learner-reached states for DAgger.
Call `teacher.act()` every step of a learner rollout so its phase follows the episode, and `teacher.reset(ids)` when an environment resets.

Measured on the batched environment, 20 episodes per object on seeds 900000 to 900019 (the same poses for every object), 200-step horizon:

| Split | Objects | Mean teacher success | Lowest |
|---|---|---|---|
| train | 27 | 0.985 | 0.80 (ladybug_bead) |
| held-out test | 14 | 0.996 | 0.95 (supplement_tub) |
| dropped | 4 | 0.29 | 0.00 (bird_rattle) |

Successful episodes take 117 to 132 steps on average, depending on the object.

## Throughput

128 batched environments on the train split, 4 simulation threads (`FLYARM_SIM_THREADS=4`), Apple M5 Max, other jobs sharing the machine: 7,350 environment steps per second with random actions (physics, IK and task rules) and 6,100 with the teacher in the loop (`docs/results/grasp-teacher.json`, `throughput`).
One control step is 25 MuJoCo steps, so that is about 180,000 physics steps per second.

## Reproduce

```
PYTHONPATH=src .venv/bin/python scripts/fetch_grasp_objects.py
FLYARM_SIM_THREADS=4 PYTHONPATH=src .venv/bin/python scripts/grasp_smoke.py
FLYARM_MODEL=assets/menagerie/franka_emika_panda/scene.xml PYTHONPATH=src .venv/bin/python -m pytest tests/test_grasp_env.py tests/test_grasp_objects.py
```

`grasp_smoke.py` writes the per-object teacher table, the throughput and the video record to `docs/results/grasp-teacher.json` and a labelled video of the teacher on four train and four held-out objects to `runs/grasp/teacher-smoke.mp4` (a camera looks along each object's long axis, so the jaws close across the view).
Rebuilding the object set from scratch is `fetch_grasp_objects.py --measure`, then `grasp_smoke.py --skip-video --skip-throughput`, then `grasp_split.py`, then `grasp_smoke.py` again.

## Known limits

- Collision uses each scan's convex hull: mugs have no hole behind the handle, bowls are solid, and the grasp is always an outside pinch.
- Objects are scaled to fit the Panda hand (most mugs and bowls to 0.3 to 0.6), so they are toy-sized versions of the real items; the scale is in the catalog.
- Mass comes from hull volume at one density, not from the real item, and is clipped to [30, 300] g; friction is the same for every object.
- Objects are placed upright in their scanned pose; a few settle a few millimetres or degrees on the first step, and the lift is measured from the declared resting height.
- The Menagerie gripper closes with only 2 to 3 N, and the position servos leave a 3 to 5 mm steady-state error on small commanded steps; the teacher tolerates both, a learned policy has to as well.
- The teacher's per-object success is measured on 20 episodes; objects near the 0.8 threshold (ladybug_bead) could fall on either side with other seeds.
- The split is by object, not by family: the held-out objects are new shapes of families seen in training, not unseen families.

## Wiring it into training

Not done here.
The imitation pipeline (`flyarm.whole_brain.experiment.Task`) needs a grasp branch: `PandaGraspEnv` with `obs_dim` 54 and `action_dim` 5, a `collect` that records `GraspTeacher` actions and phases, stage-balanced weights over the four teacher phases, a DAgger collector that steps the learner and calls `teacher.act()` for labels, and seed splits for episodes that sit on top of the object split (train episodes on `split="train"`, validation and test episodes on held-out seeds for train objects and on `split="test"` for unseen objects).
PPO needs a `TaskAdapter` like `PickPlaceTask`: `make_env` returning `BatchedGrasp(split="train", reward=RewardConfig(success_bonus=..., gamma=settings.gamma))`, `privileged_dim` 59, and a `score` that evaluates deterministic mean actions separately on seen and unseen objects; its `extra` counter can be episodes that lifted.
Controllers whose readout assumes four actions (the pick-and-place decoder) need a five-output decoder, and the connectome interface needs an encoder sized for 54 inputs.
