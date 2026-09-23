"""The two public environments: :class:`BatchedGrasp` (mjbatch) and :class:`PandaGraspEnv`.

Both compile the same model (the Panda plus every object of the chosen split) and apply the
same rules through :class:`flyarm.grasp.sim.GraspSim`; only the physics backend differs.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from flyarm.grasp import task
from flyarm.grasp.objects import GraspObject, split_objects
from flyarm.grasp.scene import CAMERA
from flyarm.grasp.sim import GraspSim, MjbatchPhysics, MjDataPhysics, build_model

DEFAULT_ASSET_ROOT = Path("assets/objects")


def resolve_objects(split: str, objects: Sequence[GraspObject] | None) -> list[GraspObject]:
    """Explicit objects win; otherwise the committed split (``train``, ``test`` or ``all``)."""
    return list(objects) if objects is not None else split_objects(split)


class BatchedGrasp(GraspSim):
    """N grasp episodes stepped in parallel by mjbatch; one batch mixes objects freely.

    API parallel to BatchedPickPlace: ``reset(ids, seeds)``, ``step(action, auto_reset)``,
    ``observation(privileged=...)``, ``obs_dim``, ``action_dim``, ``privileged_dim``, ``horizon``.
    """

    def __init__(
        self,
        model_path: Path,
        num_envs: int,
        *,
        split: str = "train",
        objects: Sequence[GraspObject] | None = None,
        asset_root: Path = DEFAULT_ASSET_ROOT,
        horizon: int = task.DEFAULT_HORIZON,
        first_seed: int = 0,
        num_threads: int = 0,
        reward: task.RewardConfig | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        chosen = resolve_objects(split, objects)
        model = build_model(Path(model_path), chosen, Path(asset_root))
        super().__init__(
            model,
            chosen,
            MjbatchPhysics(model, num_envs, num_threads),
            horizon=horizon,
            first_seed=first_seed,
            reward=reward,
        )


class PandaGraspEnv(gym.Env[np.ndarray, np.ndarray]):
    """One grasp episode in a plain MjData, with rendering, for teachers, videos and tests.

    ``reset(seed=s, options={"object": name_or_index})`` starts the same episode as
    ``BatchedGrasp.reset(seeds=[s], objects=[index])``.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        model_path: Path,
        *,
        split: str = "train",
        objects: Sequence[GraspObject] | None = None,
        asset_root: Path = DEFAULT_ASSET_ROOT,
        horizon: int = task.DEFAULT_HORIZON,
        reward: task.RewardConfig | None = None,
        render_mode: str | None = None,
        render_size: tuple[int, int] = (480, 640),
    ) -> None:
        if render_mode not in (None, "rgb_array"):
            raise ValueError("render_mode must be None or 'rgb_array'")
        chosen = resolve_objects(split, objects)
        self.model = build_model(Path(model_path), chosen, Path(asset_root))
        self.physics = MjDataPhysics(self.model)
        self.data = self.physics.data
        self.sim = GraspSim(self.model, chosen, self.physics, horizon=horizon, reward=reward)
        self.horizon, self.render_mode, self.render_size = horizon, render_mode, render_size
        self.action_space = gym.spaces.Box(-1.0, 1.0, (task.ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (task.OBS_DIM,), np.float32)
        self._renderer: mujoco.Renderer | None = None

    @property
    def objects(self) -> list[GraspObject]:
        return self.sim.objects

    def object_id(self, key: int | str) -> int:
        if isinstance(key, str):
            return self.sim.object_names.index(key)
        if not 0 <= int(key) < len(self.sim.objects):
            raise ValueError(f"object index {key} out of range")
        return int(key)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if seed is None:
            seed = int(self.np_random.integers(2**31))
        objects = None
        if "object" in options:
            objects = np.array([self.object_id(options["object"])])
        obs = self.sim.reset(np.array([0]), np.array([seed]), objects)
        return obs[0], self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.sim.step(np.asarray(action, dtype=np.float64)[None], auto_reset=False)
        return (
            result.obs[0],
            float(result.reward[0]),
            bool(result.terminated[0]),
            bool(result.truncated[0]),
            self._info(),
        )

    def _info(self) -> dict[str, Any]:
        sim = self.sim
        left, right = sim.contacts()
        gain = float(sim.height_gain()[0])
        return {
            "object": sim.object_names[int(sim.object_index[0])],
            "object_index": int(sim.object_index[0]),
            "contact_left": bool(left[0]),
            "contact_right": bool(right[0]),
            "grasped": bool(left[0] and right[0]),
            "ever_grasped": bool(sim.ever_grasped[0]),
            "ever_lifted": bool(sim.ever_lifted[0]),
            "height_gain": gain,
            "hold_steps": int(sim.hold[0]),
            "is_success": bool(task.is_success(sim.hold)[0]),
            "gripper_opening": float(sim.gripper_opening()[0]),
            "steps": int(sim.steps[0]),
        }

    def render(self, camera: str | mujoco.MjvCamera = CAMERA) -> np.ndarray:
        """RGB frame from the scene's fixed camera, or from ``camera`` (a name or free camera)."""
        if self._renderer is None:
            height, width = self.render_size
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def side_camera(self, distance: float = 0.55, elevation: float = -12.0) -> mujoco.MjvCamera:
        """A free camera looking along the object's long axis, so the jaws close across view.

        Of the two ends of the axis it looks from the one farther from the robot base, and it
        aims above the object so the whole lift stays in frame.
        """
        sim = self.sim
        position = sim.object_pos()[0]
        axis = sim.object_rotation()[0][:2, 0]
        axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
        if np.dot(axis, position[:2]) < 0:  # view from the side away from the base
            axis = -axis
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:] = [position[0], position[1], 0.14]
        camera.distance = distance
        camera.azimuth = float(np.degrees(np.arctan2(-axis[1], -axis[0])))
        camera.elevation = elevation
        return camera

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
