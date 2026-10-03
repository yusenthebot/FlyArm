"""Public environments: :class:`BatchedManipulation` (mjbatch) and :class:`PandaManipulationEnv`.

Both compile the same model for a split (the Panda, that split's objects, the furniture) and
apply the same rules through :class:`flyarm.manipulation.sim.ManipulationSim`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np

from flyarm.grasp.arm import MjbatchPhysics, MjDataPhysics
from flyarm.grasp.objects import split_objects
from flyarm.grasp.task import ACTION_DIM
from flyarm.manipulation import tasks as tk
from flyarm.manipulation.scene import CAMERA, build_manipulation_model
from flyarm.manipulation.sim import OBS_DIM, ManipulationSim, RewardConfig
from flyarm.manipulation.splits import SPLITS, Split

DEFAULT_ASSET_ROOT = Path("assets/objects")


def _split(name: str | Split) -> Split:
    return name if isinstance(name, Split) else SPLITS[name]


class BatchedManipulation(ManipulationSim):
    """N articulated multi-step episodes stepped in parallel by mjbatch.

    API parallel to BatchedPickPlace and BatchedGrasp: ``reset(ids, seeds)``,
    ``step(action, auto_reset)``, ``observation(privileged=...)``, ``obs_dim``,
    ``action_dim``, ``privileged_dim``, ``horizon`` (the longest template's).
    """

    def __init__(
        self,
        model_path: Path,
        num_envs: int,
        *,
        split: str | Split = "train",
        asset_root: Path = DEFAULT_ASSET_ROOT,
        first_seed: int | None = None,
        num_threads: int = 0,
        reward: RewardConfig | None = None,
        cue: bool = True,
        velocities: bool = True,
        phase_cue: bool = False,
        templates: tuple[str, ...] | None = None,
    ) -> None:
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        chosen = _split(split)
        self.split = chosen
        objects = split_objects(chosen.objects)
        model = build_manipulation_model(Path(model_path), objects, Path(asset_root))
        super().__init__(
            model,
            objects,
            MjbatchPhysics(model, num_envs, num_threads),
            templates=templates or chosen.templates,
            ranges=chosen.ranges,
            first_seed=chosen.seed_start if first_seed is None else first_seed,
            reward=reward,
            cue=cue,
            velocities=velocities,
            phase_cue=phase_cue,
        )


class PandaManipulationEnv(gym.Env[np.ndarray, np.ndarray]):
    """One episode in a plain MjData, with rendering, for teachers, videos and tests.

    ``reset(seed=s, options={"template": name})`` starts the same episode as
    ``BatchedManipulation.reset(seeds=[s], templates=[name])`` on the same split.
    """

    metadata: dict[str, Any] = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        model_path: Path,
        *,
        split: str | Split = "train",
        asset_root: Path = DEFAULT_ASSET_ROOT,
        reward: RewardConfig | None = None,
        cue: bool = True,
        velocities: bool = True,
        phase_cue: bool = False,
        render_size: tuple[int, int] = (480, 640),
        templates: tuple[str, ...] | None = None,
    ) -> None:
        chosen = _split(split)
        self.split = chosen
        objects = split_objects(chosen.objects)
        self.model = build_manipulation_model(Path(model_path), objects, Path(asset_root))
        self.physics = MjDataPhysics(self.model)
        self.data = self.physics.data
        self.sim = ManipulationSim(
            self.model,
            objects,
            self.physics,
            templates=templates or chosen.templates,
            ranges=chosen.ranges,
            reward=reward,
            cue=cue,
            velocities=velocities,
            phase_cue=phase_cue,
        )
        self.render_size = render_size
        self.action_space = gym.spaces.Box(-1.0, 1.0, (ACTION_DIM,), np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self._renderer: mujoco.Renderer | None = None

    @property
    def horizon(self) -> int:
        return int(self.sim.horizons[0])

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        if seed is None:
            seed = int(self.np_random.integers(2**31))
        templates = [options["template"]] if "template" in options else None
        episodes = [options["episode"]] if "episode" in options else None
        obs = self.sim.reset(np.array([0]), np.array([seed]), templates, episodes)
        return obs[0], self._info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        result = self.sim.step(np.asarray(action, dtype=np.float64)[None], auto_reset=False)
        return (
            result.obs[0],
            float(result.reward[0]),
            bool(result.terminated[0]),
            bool(result.truncated[0]),
            self._info(result),
        )

    def _info(self, result: Any = None) -> dict[str, Any]:
        sim = self.sim
        episode = sim.episodes[0]
        assert episode is not None
        state = sim.scene_state()
        done = sim.subgoal_done(sim.effects(state))
        leading = int(sim.leading(done)[0])
        names = [sim.objects[i].name if i >= 0 else "" for i in episode.objects]
        current = episode.subgoals[min(leading, len(episode.subgoals) - 1)]
        return {
            "template": episode.template,
            "subgoals": [goal.describe(names) for goal in episode.subgoals],
            "subgoals_done": leading,
            "current": current.describe(names) if leading < len(episode.subgoals) else "done",
            "is_success": leading >= len(episode.subgoals),
            "objects": names,
            "joints": sim.joints()[0].tolist(),
            "stray_contact": bool(result.stray_contact[0]) if result is not None else False,
            "steps": int(sim.steps[0]),
        }

    def render(self, camera: str | mujoco.MjvCamera = CAMERA) -> np.ndarray:
        if self._renderer is None:
            height, width = self.render_size
            # The offscreen framebuffer defaults to 640 x 480; grow it for larger renders.
            visual = self.model.vis.global_
            visual.offwidth = max(int(visual.offwidth), width)
            visual.offheight = max(int(visual.offheight), height)
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


__all__ = ["BatchedManipulation", "PandaManipulationEnv", "tk"]
