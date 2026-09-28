# Figure: FlyArm system overview

**Figure.** A frozen, complete fruit-fly central nervous system controls a simulated robot arm.
(a) One control step.
The normalized observation $o_t$ is mapped by a trained linear encoder $W_{\mathrm{in}}$ (flame) to currents $I_t$ that enter only the 1,846 ascending sensory neurons of the MaleCNS v1.0 connectome (snowflake: frozen).
The connectome has 166,700 neurons and 10.5 million synapses, and its weights are the measured synapse counts; it is never modified.
Each control step runs three neural steps $h \leftarrow 0.5\,h + 0.5\tanh(I + g W h)$ with $g = 0.8$, and the readout is the mean rate of 2,022 neurons (1,314 descending and 708 VNC motor neurons) over those three steps.
A frozen calibration standardizes the readout and divides it by $\sqrt{2022}$, and a trained linear decoder $W_{\mathrm{out}}$ followed by $\tanh$ gives the action $a_t$, which steps the MuJoCo simulation.
The neural state $h$ persists across control steps, so the controller has memory.
This whole-body interface (B1a) is used for every main result; the earlier B2 front-leg interface fed 23 front-leg proprioceptors and 4,868 head sensory neurons and read 68 front-leg motor neurons.
The network runs with MLX on the Apple GPU at about 4 ms per control step for one environment and about 20 ms for 128 environments in parallel.
On the complex manipulation benchmark, with a linear input and a linear readout, the connectome acts essentially as one fixed synaptic projection: the signal carried by deeper synaptic paths is of order $10^{-3}$, and a degree-preserving shuffle performs the same.
The render shows the anatomy of the network, not a demonstrated multi-hop computation.
(b) Tasks in MuJoCo with the Franka Panda, given as observation and action dimensions and episode length.
Pick-and-place of a block: 37 to 4 ($\Delta x$, $\Delta y$, $\Delta z$, gripper), 400 steps.
FrankaKitchen with the four D4RL subtasks microwave, kettle, light switch and slide cabinet: 30 position features to 9 joint velocities, 280 steps of 80 ms.
Generalizable grasping of 41 scanned objects: 54 to 5 ($\Delta x$, $\Delta y$, $\Delta z$, wrist rotation, gripper), about 200 steps.
Complex manipulation with a drawer cabinet, a flip cabinet and a storage box, 2 to 4 objects and 11 task templates of 3 to 8 subgoals each: 220 dimensions including the current subtask cue to 5, 1,100 to 2,600 steps.
All tasks run in mjbatch, which steps N MuJoCo copies on a C++ thread pool with per-step results identical to a single environment; the batched kitchen matches the official environment to within $5 \times 10^{-7}$.
(c) Training in three stages.
A: behaviour cloning with DAgger from a scripted teacher or a demonstration tracker, with truncated backpropagation through time through the frozen connectome, skill-balanced sample weights, and the best stage chosen by closed-loop validation.
B: PPO warm-started from the stage-A checkpoint.
The actor is the control loop in (a); it trains $W_{\mathrm{out}}$ and optionally $W_{\mathrm{in}}$ through a one-step truncated REINFORCE update.
The critic is an MLP on privileged simulator state, used only during training; the update uses GAE, PPO clipping, advantage clipping and a DAPG term, the squared error to the demonstration actions.
The reward, validated on the kitchen, stacks six parts.
Completion bonuses are paid in demonstration order.
A potential-based term pays only for progress toward the next goal, so standing still earns nothing.
Action-quality terms reward completion depth from the success line to full placement and penalize contact with non-target objects, disturbance of objects outside the task, and action magnitude and roughness.
A strict completion bonus pays half the completion bonus only within 0.1 of the goal.
Episodes start from intermediate states of a demonstration or of the scripted teacher, using states only and no actions.
A curriculum moves from single skills to two or three steps to the full task.
C: evaluation in the official environments with strict metrics, on checkpoints selected on validation seeds.
The held-out splits are kitchen starts perturbed by 0.2 and 0.3 rad, never seen in training, and, for complex manipulation, IID episodes, unseen objects, unseen furniture and unseen skill combinations.
(d) Controls on the same code path as the main controller.
Baselines replace the connectome with a degree-preserving shuffle (every neuron keeps its in-degree and out-degree) or with parameter-matched GRU and MLP networks.
Lesions of a trained policy remove all edges, keep only the direct synapses, reset the neural state at every control step, or deafferent one input channel.
Renders: pick-and-place and kitchen are crops of the live UI; the grasp objects and the manipulation scene are rendered from the environments at reset by `src/render_tasks.py`.
The brain render was supplied by the user; its source and license must be credited before publication (see `src/assets/ASSETS.md`).
