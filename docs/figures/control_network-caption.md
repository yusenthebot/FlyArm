# Figure: FlyArm control network

**Figure.** FlyArm closes a control loop through a frozen, complete fruit-fly central nervous system.
(b) The controller is the MaleCNS v1.0 connectome of the male fruit fly (brain and ventral nerve cord): 166,700 neurons and 10,520,377 connections with at least 3 synaptic contacts.
Its weights $W$ are synapse counts times transmitter sign, row-normalized, and they are never trained.
Each neural step applies $h \leftarrow 0.5\,h + 0.5\tanh(I + 0.8\,Wh)$, three neural steps make one control step, and the readout is the mean over those three steps.
The state $h$ persists across control steps and is reset only at the start of an episode.
The network runs on Apple Silicon with MLX and a custom Metal sparse kernel at about 4 ms per control step.
The soma map shows measured soma locations (gray), the declared input neurons of either interface (amber) and the declared output neurons (teal); sensory afferents, which have no soma inside the CNS, are drawn at the contact-weighted centroid of their postsynaptic partners' somata, as in the live UI.
(a) The only trained parts are linear encoders $W_{\mathrm{in}}$ (flame), which write the observation as input current $I$ into the declared input neurons only.
In B1a pick-and-place (MuJoCo Franka Panda with real contact), the simulator state $s_t$ enters the 1,846 ascending neurons.
In B2 FrankaKitchen, the arm is the fly's left front leg: the robot joint angles $q_t$ enter the 23 left front-leg proprioceptors (chordotonal organ, hair plate and campaniform sensilla of the ProLN nerve), and the kitchen object state $o_t$ enters the 4,868 head sensory neurons, each channel through its own linear map.
(c) Linear decoders $W_{\mathrm{out}}$ (flame) read the declared output neurons only, so there is no path from inputs to outputs except through the connectome.
In B1a the decoder reads the 1,314 descending and 708 VNC motor neurons and emits the end-effector action $a_t = (\Delta x, \Delta y, \Delta z, g)$, which the environment's inverse kinematics (IK) turns into joint commands.
In B2 the decoder reads the 68 left front-leg motor neurons after a frozen per-neuron readout scale $s$ (snowflake) and emits the next $k = 10$ actions at every step, each a vector of 9 joint velocities.
The executed action is ACT's temporal ensemble, a fixed weighted average of the overlapping chunks' predictions for the current step with no parameters.
A fixed $\tanh$ bounds every decoder output.
The B1a pick-and-place results use single-step actions ($k = 1$) and no readout scale; action chunking and the readout scale belong to the kitchen protocol.
Dashed lines close the loop: the action steps the MuJoCo simulation, and the next observation is read from the simulator state.
Renders are crops of the project's live UI.
(d) Every result is checked against causal controls.
Baselines replace the connectome: a degree-preserving shuffle of the whole CNS with the same interface, a parameter-matched GRU, and, in the kitchen, an MLP.
Lesions of a trained policy remove all edges, keep only the direct input-to-output synapses, silence the leg proprioceptors (deafferented leg, kitchen), or reset the neural state at every control step.
