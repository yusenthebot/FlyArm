"""Restricted-interface imitation and causal evaluation for real-contact pick-and-place."""

from __future__ import annotations

import copy
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch

from flyarm.assets import MENAGERIE_SHA, digest_file, verify_arm
from flyarm.config import PickPlaceConfig
from flyarm.experiment import save_json
from flyarm.graph import Graph, shuffle_graph
from flyarm.interfaces import NeuralInterface, canonical_interface
from flyarm.models import ActionController
from flyarm.pick_place_env import PandaPickPlaceEnv
from flyarm.pick_place_models import PickPlaceController, PickPlacePolicy

STAGES = {
    "approach": 0,
    "descend": 1,
    "close": 2,
    "lift": 3,
    "transport": 4,
    "lower": 5,
    "open": 6,
    "retreat": 7,
}


def collect_demonstrations(
    env: PandaPickPlaceEnv, seeds: list[int], path: Path
) -> dict[str, np.ndarray]:
    obs_dim = env.observation_dim
    observations = np.zeros((len(seeds), env.horizon, obs_dim), dtype=np.float32)
    actions = np.zeros((len(seeds), env.horizon, 4), dtype=np.float32)
    mask = np.zeros((len(seeds), env.horizon), dtype=np.float32)
    stages = np.full((len(seeds), env.horizon), -1, dtype=np.int16)
    success = np.zeros(len(seeds), dtype=bool)
    ever_grasped = np.zeros(len(seeds), dtype=bool)
    ever_lifted = np.zeros(len(seeds), dtype=bool)
    final_goal_error = np.zeros(len(seeds), dtype=np.float32)
    for row, seed in enumerate(seeds):
        observation, info = env.reset(seed=seed)
        for step in range(env.horizon):
            action = env.teacher_action()
            observations[row, step] = observation
            actions[row, step] = action
            mask[row, step] = 1.0
            stages[row, step] = STAGES[env.teacher_stage]
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        success[row] = bool(info["is_success"])
        ever_grasped[row] = bool(info["ever_grasped"])
        ever_lifted[row] = bool(info["ever_lifted"])
        final_goal_error[row] = float(info["goal_xy_error"])
    data: dict[str, np.ndarray] = {
        "obs": observations,
        "actions": actions,
        "mask": mask,
        "stages": stages,
        "seeds": np.asarray(seeds, dtype=np.int64),
        "teacher_success": success,
        "teacher_ever_grasped": ever_grasped,
        "teacher_ever_lifted": ever_lifted,
        "teacher_goal_xy_error": final_goal_error,
    }
    np.savez_compressed(
        path,
        obs=observations,
        actions=actions,
        mask=mask,
        stages=stages,
        seeds=data["seeds"],
        teacher_success=success,
        teacher_ever_grasped=ever_grasped,
        teacher_ever_lifted=ever_lifted,
        teacher_goal_xy_error=final_goal_error,
    )
    return data


def collect_dagger_queries(
    env: PandaPickPlaceEnv,
    policy: PickPlacePolicy | ActionController,
    seeds: list[int],
    path: Path,
) -> dict[str, np.ndarray]:
    """Label policy-visited states with the same-API teacher; never correct actions online."""
    obs_dim = env.observation_dim
    observations = np.zeros((len(seeds), env.horizon, obs_dim), dtype=np.float32)
    actions = np.zeros((len(seeds), env.horizon, 4), dtype=np.float32)
    mask = np.zeros((len(seeds), env.horizon), dtype=np.float32)
    stages = np.full((len(seeds), env.horizon), -1, dtype=np.int16)
    controller = PickPlaceController(policy) if isinstance(policy, PickPlacePolicy) else policy
    for row, seed in enumerate(seeds):
        observation, _ = env.reset(seed=seed)
        controller.reset()
        for step in range(env.horizon):
            expert_action = env.teacher_action()
            observations[row, step] = observation
            actions[row, step] = expert_action
            mask[row, step] = 1.0
            stages[row, step] = STAGES[env.teacher_stage]
            policy_action = controller.act(observation)
            observation, _, terminated, truncated, _ = env.step(policy_action)
            if terminated or truncated:
                break
    data: dict[str, np.ndarray] = {
        "obs": observations,
        "actions": actions,
        "mask": mask,
        "stages": stages,
        "seeds": np.asarray(seeds, dtype=np.int64),
    }
    np.savez_compressed(
        path,
        obs=observations,
        actions=actions,
        mask=mask,
        stages=stages,
        seeds=data["seeds"],
    )
    return data


def concatenate_data(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    keys = {"obs", "actions", "mask", "stages", "seeds"}
    return {key: np.concatenate((first[key], second[key]), axis=0) for key in keys}


def stage_balanced_weights(data: dict[str, np.ndarray]) -> np.ndarray:
    """Per-step loss weights that equalize the eight teacher phases (clipped to 0.4-4)."""
    mask = data["mask"]
    stages = data["stages"]
    valid_stages = stages[mask.astype(bool)]
    counts = np.bincount(valid_stages, minlength=len(STAGES)).astype(np.float32)
    weights = np.zeros_like(counts)
    present = counts > 0
    weights[present] = counts[present].sum() / (present.sum() * counts[present])
    weights = np.clip(weights, 0.4, 4.0)
    return mask * weights[np.maximum(stages, 0)]


def balanced_mask(data: dict[str, np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(stage_balanced_weights(data))


@torch.no_grad()
def sequence_loss(policy: PickPlacePolicy, data: dict[str, np.ndarray]) -> float:
    observations = torch.from_numpy(data["obs"])
    targets = torch.from_numpy(data["actions"])
    mask = balanced_mask(data)
    state = policy.initial_state(len(observations))
    total = torch.tensor(0.0)
    for step in range(observations.shape[1]):
        prediction, state = policy(observations[:, step], state)
        total += (((prediction - targets[:, step]) ** 2).mean(-1) * mask[:, step]).sum()
    return float(total / mask.sum())


def train_policy(
    policy: PickPlacePolicy,
    train_data: dict[str, np.ndarray],
    validation_data: dict[str, np.ndarray],
    config: PickPlaceConfig,
    seed: int,
    deadline: float,
    *,
    epochs: int | None = None,
    phase: str = "behavior_cloning",
) -> tuple[list[dict[str, float | int | str]], dict[str, float | int]]:
    observations = torch.from_numpy(train_data["obs"])
    targets = torch.from_numpy(train_data["actions"])
    masks = balanced_mask(train_data)
    samples = train_data["obs"][train_data["mask"].astype(bool)]
    policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    generator = np.random.default_rng(seed)
    best_loss, best_epoch = float("inf"), -1
    best_state = copy.deepcopy(policy.state_dict())
    curves: list[dict[str, float | int | str]] = []
    started = time.monotonic()
    epoch_count = config.epochs if epochs is None else epochs
    for epoch in range(epoch_count):
        if time.monotonic() >= deadline:
            raise TimeoutError("Pick-place budget exhausted; partial artifacts retained")
        policy.train()
        losses: list[float] = []
        order = generator.permutation(len(observations))
        for start in range(0, len(observations), config.batch_size):
            indices = order[start : start + config.batch_size]
            batch_x = observations[indices]
            batch_y = targets[indices]
            batch_mask = masks[indices]
            state = policy.initial_state(len(indices))
            for time_start in range(0, batch_x.shape[1], config.bptt_steps):
                time_stop = min(time_start + config.bptt_steps, batch_x.shape[1])
                count = batch_mask[:, time_start:time_stop].sum()
                if count == 0:
                    break
                optimizer.zero_grad()
                loss = torch.tensor(0.0)
                for step in range(time_start, time_stop):
                    prediction, state = policy(batch_x[:, step], state)
                    loss += (
                        ((prediction - batch_y[:, step]) ** 2).mean(-1) * batch_mask[:, step]
                    ).sum()
                loss /= count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                state = state.detach()
                losses.append(float(loss.detach()))
        policy.eval()
        validation_loss = sequence_loss(policy, validation_data)
        curves.append(
            {
                "epoch": epoch + 1,
                "phase": phase,
                "train_mse": float(np.mean(losses)),
                "validation_mse": validation_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if validation_loss < best_loss:
            best_loss, best_epoch = validation_loss, epoch + 1
            best_state = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    return curves, {
        "best_epoch": best_epoch,
        "best_validation_mse": best_loss,
        "training_seconds": time.monotonic() - started,
    }


def evaluate(
    env: PandaPickPlaceEnv,
    policy: PickPlacePolicy | ActionController | None,
    seeds: list[int],
    *,
    mode: str = "learned",
    reset_state_every_step: bool = False,
) -> dict[str, Any]:
    controller = PickPlaceController(policy) if isinstance(policy, PickPlacePolicy) else policy
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        observation, info = env.reset(seed=seed)
        if controller is not None:
            controller.reset()
        timings: list[float] = []
        action_changes: list[float] = []
        previous = np.zeros(4, dtype=np.float32)
        steps = 0
        for _step in range(env.horizon):
            steps += 1
            if controller is not None:
                if reset_state_every_step:
                    controller.reset()
                started = time.perf_counter()
                action = controller.act(observation)
                timings.append(time.perf_counter() - started)
            elif mode == "teacher":
                action = env.teacher_action()
            elif mode == "zero":
                action = np.zeros(4, dtype=np.float32)
            else:
                raise ValueError(f"Unknown evaluation mode: {mode}")
            action_changes.append(float(np.linalg.norm(action - previous)))
            previous = action.copy()
            observation, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "success": bool(info["is_success"]),
                "steps": steps,
                "ever_grasped": bool(info["ever_grasped"]),
                "ever_lifted": bool(info["ever_lifted"]),
                "goal_xy_error_m": float(info["goal_xy_error"]),
                "object_height_m": float(info["object_height"]),
                "object_speed_mps": float(info["object_speed"]),
                "final_stage": str(info["stage"]),
                "action_delta_mean": float(np.mean(action_changes)),
                "inference_ms_median": (float(np.median(timings) * 1000) if timings else None),
            }
        )
    return {
        "mode": mode,
        "reset_state_every_step": reset_state_every_step,
        "episodes": episodes,
        "success_rate": float(np.mean([episode["success"] for episode in episodes])),
        "grasp_rate": float(np.mean([episode["ever_grasped"] for episode in episodes])),
        "lift_rate": float(np.mean([episode["ever_lifted"] for episode in episodes])),
        "mean_goal_xy_error_m": float(
            np.mean([episode["goal_xy_error_m"] for episode in episodes])
        ),
    }


def record_rollouts(
    env: PandaPickPlaceEnv,
    policy: PickPlacePolicy,
    seeds: list[int],
    output: Path,
) -> None:
    """Record physics, contacts, actions, and hidden state from the same runs."""
    controller = PickPlaceController(policy)
    trace: list[dict[str, Any]] = []
    frame_index = 0
    with cast(Any, imageio.get_writer(output, fps=20, codec="libx264", quality=7)) as writer:
        for seed in seeds:
            controller.reset()
            observation, info = env.reset(seed=seed)
            writer.append_data(env.render())
            trace.append(
                {
                    "frame": frame_index,
                    "seed": seed,
                    "step": -1,
                    "action": [0.0] * 4,
                    "hidden": [0.0] * policy.state_size,
                    "stage": str(info["stage"]),
                    "contact_left": bool(info["contact_left"]),
                    "contact_right": bool(info["contact_right"]),
                    "object_height_m": float(info["object_height"]),
                    "goal_xy_error_m": float(info["goal_xy_error"]),
                    "success": False,
                }
            )
            frame_index += 1
            for step in range(env.horizon):
                action = controller.act(observation)
                observation, _, terminated, truncated, info = env.step(action)
                trace.append(
                    {
                        "frame": frame_index,
                        "seed": seed,
                        "step": step,
                        "action": action.tolist(),
                        "hidden": np.round(controller.state[0].numpy(), 4).tolist(),
                        "stage": str(info["stage"]),
                        "contact_left": bool(info["contact_left"]),
                        "contact_right": bool(info["contact_right"]),
                        "object_height_m": float(info["object_height"]),
                        "goal_xy_error_m": float(info["goal_xy_error"]),
                        "success": bool(info["is_success"]),
                    }
                )
                frame = env.render()
                writer.append_data(frame)
                frame_index += 1
                if terminated or truncated:
                    imageio.imwrite(output.with_suffix(".png"), frame)
                    break
    save_json(
        output.with_suffix(".json"),
        {"kind": policy.kind, "fps": 20, "seeds": seeds, "trajectory": trace},
    )


def run_pick_place_experiment(
    graph_path: Path,
    model_path: Path,
    output: Path,
    config: PickPlaceConfig,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    started = time.monotonic()
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    try:
        try:
            return _run_pick_place_experiment(graph_path, model_path, output, config)
        except (Exception, KeyboardInterrupt) as error:
            if output.is_dir():
                result_path = output / "results.json"
                partial = (
                    json.loads(result_path.read_text()) if result_path.exists() else {"models": []}
                )
                partial.update(
                    status="failed",
                    error_type=type(error).__name__,
                    error=str(error),
                    elapsed_seconds=time.monotonic() - started,
                )
                save_json(result_path, partial)
            raise
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def _run_pick_place_experiment(
    graph_path: Path,
    model_path: Path,
    output: Path,
    config: PickPlaceConfig,
) -> dict[str, Any]:
    verified_scene = verify_arm(model_path.resolve().parent.parent)
    if verified_scene.resolve() != model_path.resolve():
        raise ValueError("Model must be the pinned, unmodified Menagerie Panda scene.xml")
    graph = Graph.load(graph_path)
    graph.validate_mvp_provenance()
    interface = canonical_interface(graph)
    output.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + config.max_seconds
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    graph.save(output / "graph.npz")
    interface.save(output / "interface.json")
    save_json(output / "config.json", config.model_dump())
    save_json(
        output / "provenance.json",
        {
            "graph_sha256": digest_file(graph_path),
            "graph_content_sha256": graph.fingerprint(),
            "source_files_sha256": {
                path.name: digest_file(path) for path in sorted(Path(__file__).parent.glob("*.py"))
            },
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {name: version(name) for name in ["torch", "mujoco", "numpy", "gymnasium"]},
            "menagerie_revision": MENAGERIE_SHA,
            "observation": (
                "privileged_state_37d_with_contacts_object_velocity_and_history; no_teacher_stage"
            ),
            "action": "normalized_delta_xyz_plus_gripper",
            "task": "fixed_orientation_real_contact_cube_pick_and_place",
            "interface": interface.to_dict(),
            "interface_graph_paths": interface.path_statistics(graph),
            "rng_plan": (
                "torch model seed; numpy training seed; graph seed+17000; "
                "Gym episode seeds; fixed split ranges"
            ),
            "reproducibility_scope": (
                "seed replay on recorded platform, not cross-platform bitwise guarantee"
            ),
        },
    )
    train_seeds = list(range(40000, 40000 + config.train_episodes))
    validation_seeds = list(range(50000, 50000 + config.val_episodes))
    test_seeds = list(range(60000, 60000 + config.test_episodes))
    env = PandaPickPlaceEnv(model_path=model_path, horizon=config.horizon, render_mode="rgb_array")
    try:
        train_data = collect_demonstrations(env, train_seeds, output / "train.npz")
        validation_data = collect_demonstrations(env, validation_seeds, output / "validation.npz")
        save_json(
            output / "splits.json",
            {
                "train": train_seeds,
                "validation": validation_seeds,
                "test": test_seeds,
                "development_regression": list(range(12)),
                "protocol": "disjoint complete episodes; development seeds excluded",
            },
        )
        results: dict[str, Any] = {
            "status": "running",
            "claim": (
                "restricted graph-mediated control test; no MaleCNS topology advantage assumed"
            ),
            "teacher_train_success": float(train_data["teacher_success"].mean()),
            "teacher_validation_success": float(validation_data["teacher_success"].mean()),
            "teacher": evaluate(env, None, test_seeds, mode="teacher"),
            "zero": evaluate(env, None, test_seeds, mode="zero"),
            "models": [],
        }
        save_json(output / "results.json", results)
        if results["teacher"]["success_rate"] < 0.9:
            raise RuntimeError("Teacher success below 90%; model comparison is invalid")
        for seed in config.seeds:
            shuffled = shuffle_graph(graph, seed + 17000)
            shuffled.save(output / f"shuffled-{seed}.npz")
            for kind in config.policies:
                run = output / f"{kind}-{seed}"
                run.mkdir()
                policy_graph = shuffled if kind == "restricted_shuffled" else graph
                policy_interface = (
                    NeuralInterface.bind(
                        shuffled,
                        interface.input_body_ids,
                        interface.output_body_ids,
                        label=interface.label,
                    )
                    if kind == "restricted_shuffled"
                    else interface
                )
                policy = PickPlacePolicy(
                    kind,
                    policy_graph,
                    policy_interface,
                    seed=seed,
                    obs_dim=env.observation_dim,
                    internal_steps=config.internal_steps,
                )
                curves, training = train_policy(
                    policy, train_data, validation_data, config, seed, deadline
                )
                aggregate_data = train_data
                training_phases: list[dict[str, float | int | str]] = [
                    {"phase": "behavior_cloning", **training}
                ]
                dagger_seed_sets: list[list[int]] = []
                for iteration in range(config.dagger_iterations):
                    dagger_start = 70000 + iteration * 1000
                    dagger_seeds = list(range(dagger_start, dagger_start + config.dagger_episodes))
                    dagger_seed_sets.append(dagger_seeds)
                    dagger_data = collect_dagger_queries(
                        env,
                        policy,
                        dagger_seeds,
                        run / f"dagger-{iteration + 1}.npz",
                    )
                    aggregate_data = concatenate_data(aggregate_data, dagger_data)
                    phase = f"dagger_{iteration + 1}"
                    phase_curves, phase_training = train_policy(
                        policy,
                        aggregate_data,
                        validation_data,
                        config,
                        seed + iteration + 1000,
                        deadline,
                        epochs=config.dagger_epochs,
                        phase=phase,
                    )
                    curves.extend(phase_curves)
                    training_phases.append({"phase": phase, **phase_training})
                    training = phase_training
                save_json(run / "learning.json", curves)
                torch.save(policy.state_dict(), run / "policy.pt")
                item: dict[str, Any] = {
                    "kind": kind,
                    "seed": seed,
                    "trainable_parameters": policy.trainable_parameters(),
                    "state_size": policy.state_size,
                    **training,
                    "training_seconds": float(
                        sum(float(phase["training_seconds"]) for phase in training_phases)
                    ),
                    "training_phases": training_phases,
                    "dagger_seed_sets": dagger_seed_sets,
                    "clean": evaluate(env, policy, test_seeds),
                }
                if kind == "restricted_shuffled":
                    item["interface_graph_paths"] = policy_interface.path_statistics(shuffled)
                if kind == "restricted_connectome":
                    disconnected = PickPlacePolicy(
                        "restricted_disconnected",
                        graph,
                        interface,
                        seed=seed,
                        obs_dim=env.observation_dim,
                        internal_steps=config.internal_steps,
                    )
                    state = {
                        key: value
                        for key, value in policy.state_dict().items()
                        if key != "adjacency"
                    }
                    disconnected.load_state_dict(state, strict=False)
                    item["edges_silenced"] = evaluate(env, disconnected, test_seeds)
                    item["state_reset_every_step"] = evaluate(
                        env, policy, test_seeds, reset_state_every_step=True
                    )
                save_json(run / "evaluation.json", item)
                results["models"].append(item)
                save_json(output / "results.json", results)
                print(
                    f"{kind} seed={seed} val={training['best_validation_mse']:.5f} "
                    f"grasp={item['clean']['grasp_rate']:.1%} "
                    f"place={item['clean']['success_rate']:.1%}",
                    flush=True,
                )
                if seed == config.seeds[0] and kind == "restricted_connectome":
                    record_rollouts(env, policy, test_seeds[:3], output / "policy-rollout.mp4")
        results.update(status="complete", elapsed_seconds=time.monotonic() - started)
        save_json(output / "results.json", results)
        return results
    finally:
        env.close()
