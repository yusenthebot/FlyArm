# Figure: FlyArm pipeline

**Figure.** One control step of FlyArm: a frozen, complete fruit-fly central nervous system controls a simulated Franka arm.
(a) The observation $o_t$ is simulator state: the arm $q_t$, the objects $x_t$, the scene $s_t$ and, where the task has one, a task cue $c_t$.
On the articulated manipulation benchmark it has 230 values (arm 21, four object slots of 22, furniture 46, and a cue of 75 that names the current skill, object, receptacle and observable motor phase), with joint and object velocities zeroed and 70 fixed fine-scale copies of the hand-relative offsets appended; on FrankaKitchen it is the benchmark's 30 position features.
(b) The trained encoder $E_\phi$ writes input currents into the 1,846 ascending neurons only: a two-layer tanh MLP of width 256 on the manipulation benchmark, a linear map on FrankaKitchen.
(c) The MaleCNS v1.0 connectome (brain and ventral nerve cord: 166,700 neurons, 10.5 million connections with at least three synaptic contacts) is never trained; its weights $W$ are signed, row-normalized synapse counts, and each control step runs three neural steps of $h \leftarrow 0.5\,h + 0.5\tanh(I + 0.8\,Wh)$.
The soma map shows measured soma locations (gray), declared input neurons (amber) and output neurons (teal).
(d) The 1,314 descending and 708 VNC motor neurons are centred and scaled by a readout normalization fixed from demonstrations, and the trained linear decoder $W_{\mathrm{out}}$ with a $\tanh$ maps them to the action $a_t$: an end-effector step $(\Delta x, \Delta y, \Delta z, \Delta\psi, g)$ executed by inverse kinematics on the manipulation benchmark, and 9 joint-velocity commands on FrankaKitchen.
(e) MuJoCo advances the physics and the dashed line returns the next state, closing the loop at 20 Hz on the manipulation benchmark.
The renders are a frame of the reported checkpoint solving an iid test episode of the manipulation benchmark and the FrankaKitchen scene.
(f) Only $E_\phi$ and $W_{\mathrm{out}}$ are trained: imitation of demonstrations with DAgger, then PPO with a demonstration term (DAPG); on the manipulation benchmark PPO also updates the encoder through one control step of the frozen connectome.
