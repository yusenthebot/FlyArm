"""Live UI for B2: a trained front-leg fly controller drives the Franka in FrankaKitchen.

One worker thread owns MLX (whose GPU streams are per-thread) and the MuJoCo kitchen; HTTP
handlers enqueue commands and read the snapshot the worker publishes after every change.
Modes switch between the trained measured-CNS policy, the trained shuffled-CNS policy and
lesions of the measured policy (edges off, direct synapses only, deafferented leg, head
senses removed); each switch restarts the same episode.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np

from flyarm.benchmarks import kitchen
from flyarm.flyleg.interface import front_leg_interface
from flyarm.flyleg.record import load_flyleg_policy
from flyarm.live import (
    ACTIVITY_FLOOR,
    CausalMode,
    ResetRequest,
    body_poses,
    build_scene_payload,
    create_app,
    encode_activity,
)
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.diagnostics import direct_only_weights
from flyarm.whole_brain.live import NeuronDetails, build_whole_brain_payload
from flyarm.whole_brain.policy import BrainPolicy, MlxController

BONUS_THRESHOLD = 0.3  # FrankaKitchen's own task-completion distance


class KitchenRuntime:
    def __init__(
        self,
        env: Any,
        controllers: dict[CausalMode, MlxController],
        evidence: dict[str, Any],
        body_ids: list[int],
        *,
        seed: int,
    ) -> None:
        self.env = env
        self.controllers = controllers
        self.evidence = evidence
        self.body_ids = body_ids
        self.seed = seed
        self.mode: CausalMode = "connectome"
        self.running = False
        self._commands: queue.Queue[tuple[Callable[[], None], threading.Event, list[Exception]]]
        self._commands = queue.Queue()
        self._published: dict[str, Any] = {}
        self._lock = threading.Lock()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._observation: dict[str, Any] = {}
        self._completed: list[str] = []
        self._action = np.zeros(kitchen.ACTION_DIM, dtype=np.float32)
        self._step = 0

    # Worker-thread operations ------------------------------------------------------------
    def _restart(self) -> None:
        self._observation, _ = self.env.reset(seed=self.seed)
        for controller in self.controllers.values():
            controller.reset()
        self._completed, self._step = [], 0
        self._action = np.zeros(kitchen.ACTION_DIM, dtype=np.float32)

    def _advance(self) -> None:
        controller = self.controllers[self.mode]
        features = np.asarray(self._observation["observation"], dtype=np.float32)
        self._action = controller.act(features[kitchen.POLICY_FEATURES])
        self._observation, _, terminated, truncated, info = self.env.step(
            self._action.astype(np.float64)
        )
        self._completed = list(info["episode_task_completions"])
        self._step += 1
        if terminated or truncated:
            self.running = False

    def _snapshot(self) -> dict[str, Any]:
        controller = self.controllers[self.mode]
        hidden = np.asarray(controller.state)[:, 0]
        plan = controller.plan
        goals = self.env.unwrapped.goal
        achieved = self._observation.get("achieved_goal", {})
        distances = {
            task: float(np.linalg.norm(np.asarray(achieved[task]) - np.asarray(goal)))
            for task, goal in goals.items()
            if task in achieved
        }
        total = len(goals)
        done = len(self._completed)
        model_dt = float(self.env.unwrapped.robot_env.dt)
        return {
            "connected": True,
            "running": self.running,
            "mode": self.mode,
            "step": self._step,
            "sim_time": self._step * model_dt,
            "control_dt": model_dt,
            "stage": f"{done}/{total} tasks",
            "success": done == total,
            "tasks": list(goals),
            "completed": list(self._completed),
            "task_progress": {
                task: float(np.clip(1 - distance / 1.5, 0, 1))
                if task not in self._completed
                else 1.0
                for task, distance in distances.items()
            },
            "task_distance": distances,
            "task_threshold": BONUS_THRESHOLD,
            "score": 25.0 * done,
            # Pick-and-place fields kept neutral so shared UI components stay well-defined.
            "contact_left": False,
            "contact_right": False,
            "ever_grasped": False,
            "ever_lifted": False,
            "object_height": done / total,
            "goal_error": 0.0,
            "gripper_opening": 0.0,
            "object": [0.0, 0.0, 0.0],
            "goal": [0.0, 0.0, 0.0],
            "end_effector": [0.0, 0.0, 0.0],
            "robot_points": [],
            "gripper_points": [],
            "action": self._action.tolist(),
            # Chunked controllers: the newest predicted chunk [k, 9], rows control_dt apart.
            "plan": plan.tolist() if plan is not None and len(plan) > 1 else None,
            "robot_bodies": body_poses(self.env.unwrapped.data, self.body_ids),
            "hidden_q": encode_activity(hidden),
            "hidden_count": int(hidden.size),
            "hidden_floor": ACTIVITY_FLOOR,
            "evidence": self.evidence,
        }

    def _publish(self) -> None:
        snapshot = self._snapshot()
        with self._lock:
            self._published = snapshot

    def _loop(self) -> None:
        period = 1.0 / float(self.env.metadata.get("render_fps", 12.5))
        with mx.stream(mx.new_stream(mx.gpu)):
            self._restart()
            self._publish()
            while not self._shutdown.is_set():
                started = time.monotonic()
                while True:
                    try:
                        operation, done, error = self._commands.get_nowait()
                    except queue.Empty:
                        break
                    try:
                        operation()
                    except Exception as exception:  # returned to the HTTP caller
                        error.append(exception)
                    finally:
                        self._publish()
                        done.set()
                if self.running:
                    self._advance()
                    self._publish()
                self._shutdown.wait(max(period - (time.monotonic() - started), 0.001))

    # HTTP-thread API --------------------------------------------------------------------
    def _submit(self, operation: Callable[[], None]) -> None:
        if self._thread is None:
            raise RuntimeError("Runtime worker is not running")
        done, error = threading.Event(), cast(list[Exception], [])
        self._commands.put((operation, done, error))
        if not done.wait(timeout=10.0):
            raise TimeoutError("Runtime worker did not respond")
        if error:
            raise error[0]

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="flyleg-live", daemon=True)
            self._thread.start()

    def close(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.env.close()

    def run(self) -> None:
        self._submit(lambda: setattr(self, "running", True))

    def pause(self) -> None:
        self._submit(lambda: setattr(self, "running", False))

    def step_once(self) -> None:
        def operation() -> None:
            self.running = False
            self._advance()

        self._submit(operation)

    def reset(self, request: ResetRequest | None = None) -> None:
        if request is not None and (request.object is not None or request.goal is not None):
            raise ValueError("The kitchen scene is fixed; only a plain reset is supported")

        def operation() -> None:
            self.running = False
            self._restart()

        self._submit(operation)

    def set_mode(self, mode: CausalMode) -> None:
        if mode not in self.controllers:
            raise ValueError(f"Mode {mode} is not available for this run")

        def operation() -> None:
            self.mode = mode
            self.running = False
            self._restart()

        self._submit(operation)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._published)


def summarize_kitchen_evidence(results: dict[str, Any] | None) -> dict[str, Any]:
    """B2 reading rules on seed-averaged normalized scores (shown as fractions of 100)."""
    summary: dict[str, Any] = {
        "graph_mediated": "pending",
        "topology_advantage": "pending",
        "connectome_success_rate": None,
        "shuffled_success_rate": None,
        "edges_off_success_rate": None,
        "direct_only_success_rate": None,
        "seeds": 0,
    }
    if not results or "summary" not in results:
        return summary
    fly = results["summary"].get("flyleg")
    shuffled = results["summary"].get("flyleg_shuffled")
    if fly is None:
        return summary
    lesions = fly.get("lesions", {})
    summary.update(
        connectome_success_rate=fly["clean"] / 100,
        shuffled_success_rate=shuffled["clean"] / 100 if shuffled else None,
        edges_off_success_rate=lesions.get("edges_off", 0.0) / 100 if lesions else None,
        direct_only_success_rate=lesions.get("direct_only", 0.0) / 100 if lesions else None,
        seeds=len(fly["seeds"]),
    )
    complete = results.get("status") == "complete"
    if lesions:
        if fly["clean"] >= 50 and lesions.get("edges_off", 100) < 5:
            summary["graph_mediated"] = (
                "supported" if complete and len(fly["seeds"]) >= 3 else "pending"
            )
        elif complete:
            summary["graph_mediated"] = "not_supported"
    if shuffled and complete and (fly["clean"] < 50 or fly["clean"] - shuffled["clean"] < 20):
        summary["topology_advantage"] = "not_supported"
    return summary


def serve_flyleg(
    run_root: Path,
    seed: int,
    pack_root: Path,
    annotations: Path,
    ui_dist: Path,
    *,
    host: str,
    port: int,
    episode: int,
) -> None:
    import uvicorn

    from flyarm.live import _validate_server_scope

    ui_dist = _validate_server_scope(ui_dist, host)
    pack = ConnectomePack.load(pack_root)
    pack.validate_b1a_provenance()
    config = json.loads((run_root / "config.json").read_text())
    include_head = config.get("sensory_channels", "proprioception+head") == "proprioception+head"
    leg = front_leg_interface(pack, annotations, include_head=include_head)
    gain = float(config.get("recurrent_gain", 0.8))
    policy = load_flyleg_policy(run_root, "flyleg", seed, pack_root, annotations)
    if not isinstance(policy, BrainPolicy):
        raise ValueError("Expected a front-leg brain checkpoint")
    variants: dict[CausalMode, BrainPolicy] = {
        "connectome": policy,
        "edges_off": policy.with_dynamics(
            "edges_off", RateDynamics(pack, leg.interface, recurrent_gain=gain, edges=False)
        ),
        "direct_only": policy.with_dynamics(
            "direct_only",
            RateDynamics(
                pack,
                leg.interface,
                recurrent_gain=gain,
                weights=direct_only_weights(pack, leg.interface),
            ),
        ),
        "deafferented": policy.silence_channel("deafferented", 0),
    }
    if include_head:
        variants["head_deprived"] = policy.silence_channel("head_deprived", 1)
    if (run_root / f"flyleg_shuffled-{seed}" / "policy.safetensors").is_file():
        shuffled = load_flyleg_policy(run_root, "flyleg_shuffled", seed, pack_root, annotations)
        if isinstance(shuffled, BrainPolicy):
            variants["shuffled"] = shuffled
    for variant in variants.values():
        mx.eval(variant.parameters())
    controllers = {mode: MlxController(variant) for mode, variant in variants.items()}
    order = ("connectome", "shuffled", "edges_off", "direct_only", "deafferented", "head_deprived")
    modes = [mode for mode in order if mode in controllers]
    payload, drawn = build_whole_brain_payload(
        pack,
        leg.interface,
        annotations,
        modes,
        input_groups=[(leg.proprioceptors, "front-leg proprioceptor")]
        + ([(leg.exteroceptors, "head sensory")] if include_head else []),
        output_label="front-leg motor neuron",
        place_afferents=True,
        task="kitchen",
    )
    payload["title"] = f"complete CNS · left front leg · {pack.nodes:,} neurons"
    env = kitchen.recover_env(config["split"])
    model = env.unwrapped.model
    body_ids = sorted(
        {int(model.geom_bodyid[g]) for g in range(model.ngeom) if model.geom_group[g] <= 2}
    )
    scene = build_scene_payload(
        model, body_ids, source="Gymnasium-Robotics FrankaKitchen-v1 (MIT), D4RL kitchen assets"
    )
    results_path = run_root / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else None
    runtime = KitchenRuntime(
        env, controllers, summarize_kitchen_evidence(results), body_ids, seed=episode
    )
    runtime.start()
    app = create_app(
        cast(Any, runtime),
        payload,
        ui_dist,
        neuron_details=NeuronDetails(pack, leg.interface, drawn),
        robot_payload=scene,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
