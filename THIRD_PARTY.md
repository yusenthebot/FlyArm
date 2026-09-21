# Reused work and attribution

FlyArm code is MIT. Upstream data, models, packages and papers keep their own terms.
No upstream connectome export or Menagerie mesh is committed to this repository.

| Resource | Role here | Terms / source |
|---|---|---|
| MaleCNS v1.0, Janelia / Google / Cambridge / MRC collaboration | Real neuron annotations, contacts and NT predictions; downloaded and hash-pinned | CC BY 4.0; [official download and citation instructions](https://male-cns.janelia.org/download/) |
| Google DeepMind MuJoCo Menagerie, Franka Emika Panda | Reused robot geometry, physics and actuators | Panda directory Apache 2.0; [source](https://github.com/google-deepmind/mujoco_menagerie/tree/822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7/franka_emika_panda) |
| MuJoCo | Physics and DLS Jacobians; no custom physics engine | Apache 2.0; [source](https://github.com/google-deepmind/mujoco) |
| Gymnasium | Standard reset/step spaces and seeded episode interface | MIT; [source](https://github.com/Farama-Foundation/Gymnasium) |
| PyTorch | Autograd, sparse recurrence, standard MLP/GRU and optimizer | BSD-style; [source](https://github.com/pytorch/pytorch) |

Menagerie is pinned to `822c2d8f877dd166c5b7d3c9f7e3c3b6589473b7`; the asset
checkout retains its upstream LICENSE files. Dataset object generations, byte sizes,
transport MD5 and SHA256 pins live in `src/flyarm/assets.py`. A download manifest is
recorded alongside the local data and embedded in the derived graph metadata.
When redistributing derived graphs, retain the MaleCNS attribution, license link,
source version and transformation metadata. FlyArm's MIT license does not override it.

## Method references, not currently integrated code

- [FlyGM](https://arxiv.org/abs/2602.17997): connectome-informed embodied control.
- [FLYNN](https://arxiv.org/abs/2607.00025): connectome-derived computational models.
- [Shiu et al., 2024](https://www.nature.com/articles/s41586-024-07763-9): whole-brain
  modeling from connectome structure. FlyArm's tanh dynamics are not their model.
- [drosophila-brain-mlx](https://github.com/Kisame76/drosophila-brain-mlx): potential
  Apple Silicon simulation backend; not a dependency or port in this MVP.
- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3): reusable future RL
  option; current training is behavior cloning, not PPO or SAC.

We reuse applicable arm assets and mature numerical/learning tools directly.
Fly-body locomotion environments are not relabeled as arm environments, and research
papers are not described as incorporated source code. These references do not supply
evidence that a fruit-fly connectome already controls a real Panda or improves on MLP.
