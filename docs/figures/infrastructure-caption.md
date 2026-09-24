# Supplementary figure: FlyArm infrastructure

**Supplementary figure.** Data, build, runs, monitoring and records of FlyArm.
(a) Upstream data under their own terms.
MaleCNS v1.0 (Janelia, Google, Cambridge and MRC) is CC BY 4.0.
The MuJoCo Menagerie Franka Panda is Apache 2.0.
The FrankaKitchen scene comes with Gymnasium-Robotics 1.4.2; its scene files are Apache 2.0 and the package is MIT.
Google Scanned Objects meshes and textures are CC BY 4.0, and their MJCF conversion is MIT.
(b) The connectome export is compiled into a connectome pack that keeps the edges with at least three synaptic contacts (166,700 neurons, 10.5 million edges).
Robot scenes and objects are downloaded at pinned upstream versions with SHA-256 hashes (Menagerie commit 822c2d8, scanned objects commit 6ff8d27, the kitchen from the Gymnasium-Robotics 1.4.2 package) and are not committed.
(c) Every training or evaluation run writes one directory with its config, results, curves, evaluations, checkpoints and latest progress clip.
The controller runs on MLX on the Apple GPU, and the simulation is batched by mjbatch.
(d) A local dashboard (FastAPI, port 8780) reads the run directories and shows each run's curves, evaluations, log and rollouts.
Its Live tab plays the newest progress clip; each run keeps exactly one clip, which is overwritten as training proceeds.
The screenshot shows the kitchen PPO run `ppo-kitchen-quality-demoreset-001`, and the frame is from that run's clip at iteration 600.
(e) The research log (entries E1 to E55 with a claims ledger), the 35 result files in `docs/results/` and the milestone document are kept in git.
