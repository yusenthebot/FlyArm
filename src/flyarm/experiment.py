"""Actual MuJoCo demonstration collection, sequence imitation, and closed-loop evaluation."""

from __future__ import annotations

import copy
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, cast

import imageio.v2 as imageio
import numpy as np
import torch

from flyarm.assets import MENAGERIE_SHA, digest_file, verify_arm
from flyarm.config import ExperimentConfig
from flyarm.env import PandaReachEnv
from flyarm.graph import Graph, shuffle_graph
from flyarm.models import Controller, Policy


def save_json(path: Path, value: dict | list) -> None:
    serialized = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as file:
        file.write(serialized)
        temporary = Path(file.name)
    temporary.replace(path)


def collect(env: PandaReachEnv, seeds: list[int], path: Path) -> dict:
    observations = np.zeros((len(seeds), env.horizon, 20), dtype=np.float32)
    actions = np.zeros((len(seeds), env.horizon, 3), dtype=np.float32)
    mask = np.zeros((len(seeds), env.horizon), dtype=np.float32)
    targets, success = [], []
    for row, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        targets.append(info["target"])
        for _step in range(env.horizon):
            action = env.teacher_action()
            observations[row, _step], actions[row, _step], mask[row, _step] = obs, action, 1
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        success.append(bool(info["is_success"]))
    data = {
        "obs": observations,
        "actions": actions,
        "mask": mask,
        "seeds": np.array(seeds),
        "targets": np.array(targets),
        "teacher_success": np.array(success),
    }
    np.savez_compressed(
        path,
        obs=observations,
        actions=actions,
        mask=mask,
        seeds=data["seeds"],
        targets=data["targets"],
        teacher_success=data["teacher_success"],
    )
    return data


@torch.no_grad()
def sequence_loss(policy: Policy, data: dict) -> float:
    obs = torch.from_numpy(data["obs"])
    target = torch.from_numpy(data["actions"])
    mask = torch.from_numpy(data["mask"])
    state = policy.initial_state(len(obs))
    total = torch.tensor(0.0)
    for t in range(obs.shape[1]):
        pred, state = policy(obs[:, t], state)
        total += (((pred - target[:, t]) ** 2).mean(-1) * mask[:, t]).sum()
    return float(total / mask.sum())


def train(
    policy: Policy,
    train_data: dict,
    val_data: dict,
    config: ExperimentConfig,
    seed: int,
    deadline: float,
) -> tuple[list[dict], dict]:
    obs = torch.from_numpy(train_data["obs"])
    targets = torch.from_numpy(train_data["actions"])
    masks = torch.from_numpy(train_data["mask"])
    samples = train_data["obs"][train_data["mask"].astype(bool)]
    policy.set_normalization(samples.mean(0), np.maximum(samples.std(0), 0.05))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    rng = np.random.default_rng(seed)
    best_loss, best_epoch = float("inf"), -1
    best_state = copy.deepcopy(policy.state_dict())
    curves = []
    started = time.monotonic()
    for epoch in range(config.epochs):
        if time.monotonic() >= deadline:
            raise TimeoutError("Experiment budget exhausted; partial artifacts retained")
        policy.train()
        losses = []
        order = rng.permutation(len(obs))
        for start in range(0, len(obs), config.batch_size):
            indices = order[start : start + config.batch_size]
            x, y, mask = obs[indices], targets[indices], masks[indices]
            state = policy.initial_state(len(indices))
            for t0 in range(0, x.shape[1], config.bptt_steps):
                count = mask[:, t0 : t0 + config.bptt_steps].sum()
                if count == 0:
                    break
                optimizer.zero_grad()
                loss = torch.tensor(0.0)
                for t in range(t0, min(t0 + config.bptt_steps, x.shape[1])):
                    prediction, state = policy(x[:, t], state)
                    loss = loss + (((prediction - y[:, t]) ** 2).mean(-1) * mask[:, t]).sum()
                loss = loss / count
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                state = state.detach()
                losses.append(float(loss.detach()))
        policy.eval()
        val_loss = sequence_loss(policy, val_data)
        curves.append(
            {
                "epoch": epoch + 1,
                "train_mse": float(np.mean(losses)),
                "val_mse": val_loss,
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch + 1
            best_state = copy.deepcopy(policy.state_dict())
    policy.load_state_dict(best_state)
    return curves, {
        "best_epoch": best_epoch,
        "best_val_mse": best_loss,
        "training_seconds": time.monotonic() - started,
    }


def evaluate(
    env: PandaReachEnv,
    policy: Policy | None,
    seeds: list[int],
    mode: str = "learned",
    noise_std: float = 0.0,
) -> dict:
    controller = Controller(policy) if policy is not None else None
    episodes = []
    for seed in seeds:
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 987654)
        if controller:
            controller.reset()
        timings, changes = [], []
        previous = np.zeros(3)
        for _step in range(env.horizon):
            noisy_obs = obs.copy()
            # Noise applies to xyz observations (metres), not velocities or joint radians.
            noisy_obs[14:] += rng.normal(0, noise_std, 6).astype(np.float32)
            start = time.perf_counter()
            if controller is not None:
                action = controller.act(noisy_obs)
            elif mode == "teacher":
                action = env.teacher_action()
            elif mode == "zero":
                action = np.zeros(3, dtype=np.float32)
            else:
                raise ValueError(mode)
            timings.append(time.perf_counter() - start)
            changes.append(float(np.linalg.norm(action - previous)))
            previous = action.copy()
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        episodes.append(
            {
                "seed": seed,
                "success": bool(info["is_success"]),
                "steps": _step + 1,
                "final_distance_m": float(info["distance"]),
                "target": np.asarray(info["target"]).tolist(),
                "action_delta_mean": float(np.mean(changes)),
                "inference_ms_median": float(np.median(timings) * 1000),
                "inference_ms_p95": float(np.quantile(timings, 0.95) * 1000),
            }
        )
    return {
        "mode": mode,
        "xyz_noise_std_m": noise_std,
        "episodes": episodes,
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_final_distance_m": float(np.mean([e["final_distance_m"] for e in episodes])),
    }


def record(env: PandaReachEnv, policy: Policy, seed: int, output: Path, episodes: int = 1) -> None:
    """Video and hidden-state traces come from the same actual closed-loop runs."""
    controller = Controller(policy)
    rows = []
    frame_index = 0
    with cast(Any, imageio.get_writer(output, fps=20, codec="libx264", quality=7)) as writer:
        for episode_seed in range(seed, seed + episodes):
            controller.reset()
            obs, info = env.reset(seed=episode_seed)
            writer.append_data(env.render())
            rows.append(
                {
                    "frame": frame_index,
                    "episode_seed": episode_seed,
                    "step": -1,
                    "action": [0.0, 0.0, 0.0],
                    "distance_m": float(info["distance"]),
                    "success": False,
                    "hidden": [0.0] * policy.state_size,
                }
            )
            frame_index += 1
            for t in range(env.horizon):
                action = controller.act(obs)
                obs, _, done, truncated, info = env.step(action)
                rows.append(
                    {
                        "frame": frame_index,
                        "episode_seed": episode_seed,
                        "step": t,
                        "action": action.tolist(),
                        "distance_m": float(info["distance"]),
                        "ee": np.asarray(info["ee_position"]).tolist(),
                        "success": bool(info["is_success"]),
                        "hidden": np.round(controller.state[0].numpy().astype(float), 4).tolist(),
                    }
                )
                frame = env.render()
                writer.append_data(frame)
                frame_index += 1
                if done or truncated:
                    imageio.imwrite(output.with_suffix(".png"), frame)
                    break
    save_json(
        output.with_suffix(".json"),
        {"seed": seed, "kind": policy.kind, "episodes": episodes, "fps": 20, "trajectory": rows},
    )


def run_experiment(
    graph_path: Path, model_path: Path, output: Path, config: ExperimentConfig
) -> dict:
    """Never overwrite a run; mark partial failures explicitly before propagating."""
    if output.exists():
        raise FileExistsError(f"Run directory already exists; choose a new output: {output}")
    started = time.monotonic()
    try:
        return _run_experiment(graph_path, model_path, output, config)
    except (Exception, KeyboardInterrupt) as error:
        if output.is_dir():
            path = output / "results.json"
            partial = json.loads(path.read_text()) if path.exists() else {"models": []}
            partial.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                elapsed_seconds=time.monotonic() - started,
            )
            save_json(path, partial)
        raise


def _run_experiment(
    graph_path: Path, model_path: Path, output: Path, config: ExperimentConfig
) -> dict:
    verified_scene = verify_arm(model_path.resolve().parent.parent)
    if verified_scene.resolve() != model_path.resolve():
        raise ValueError("Model must be the pinned, unmodified Menagerie Panda scene.xml")
    graph = Graph.load(graph_path)
    graph.validate_mvp_provenance()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config.max_seconds
    torch.set_num_threads(config.threads)
    torch.use_deterministic_algorithms(True)
    graph.save(output / "graph.npz")
    save_json(output / "config.json", config.model_dump())
    save_json(
        output / "provenance.json",
        {
            "graph_sha256": digest_file(graph_path),
            "graph_content_sha256": graph.fingerprint(),
            "source_files_sha256": {
                p.name: digest_file(p) for p in sorted(Path(__file__).parent.glob("*.py"))
            },
            "rng_plan": (
                "torch model seed; numpy Generator training seed; graph seed+7000; "
                "noise episode+987654; Gym episode seeds"
            ),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "reproducibility_scope": (
                "seed replay on recorded platform, not cross-platform bitwise guarantee"
            ),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "versions": {name: version(name) for name in ["torch", "mujoco", "numpy", "gymnasium"]},
            "menagerie_revision": MENAGERIE_SHA,
            "observation": "privileged_state_20d",
            "action": "normalized_3d_cartesian_displacement",
            "task": "fixed_orientation_reaching_no_grasping",
            "graph": graph.metadata,
        },
    )
    train_seeds = list(range(config.train_episodes))
    val_seeds = list(range(10000, 10000 + config.val_episodes))
    # The 20000-series exposed the initial IK frame bug and is now a regression set.
    test_seeds = list(range(30000, 30000 + config.test_episodes))
    env = PandaReachEnv(model_path=model_path, horizon=config.horizon, render_mode="rgb_array")
    try:
        train_data = collect(env, train_seeds, output / "train.npz")
        val_data = collect(env, val_seeds, output / "validation.npz")
        save_json(
            output / "splits.json",
            {
                "train": train_seeds,
                "validation": val_seeds,
                "test": test_seeds,
                "protocol": "disjoint seeded episodes; same target distribution",
            },
        )
        results: dict[str, Any] = {
            "status": "running",
            "claim": "pipeline demonstration, no connectome advantage established",
            "teacher_train_success": float(train_data["teacher_success"].mean()),
            "teacher": evaluate(env, None, test_seeds, mode="teacher"),
            "zero": evaluate(env, None, test_seeds, mode="zero"),
            "models": [],
        }
        save_json(output / "results.json", results)
        for seed in config.seeds:
            shuffled = shuffle_graph(graph, seed + 7000)
            shuffled.save(output / f"shuffled-{seed}.npz")
            for kind in config.policies:
                run = output / f"{kind}-{seed}"
                run.mkdir()
                policy_graph = shuffled if kind == "shuffled" else graph
                policy = Policy(kind, policy_graph, seed)
                curves, metrics = train(policy, train_data, val_data, config, seed, deadline)
                save_json(run / "learning.json", curves)
                torch.save(policy.state_dict(), run / "policy.pt")
                item = {
                    "kind": kind,
                    "seed": seed,
                    "trainable_parameters": policy.trainable_parameters(),
                    "state_size": policy.state_size,
                    **metrics,
                    "clean": evaluate(env, policy, test_seeds),
                    "noisy": evaluate(env, policy, test_seeds, noise_std=0.01),
                }
                if kind == "connectome":
                    disconnected = Policy("disconnected", graph, seed)
                    state = {k: v for k, v in policy.state_dict().items() if k != "adjacency"}
                    disconnected.load_state_dict(state, strict=False)
                    item["edges_silenced"] = evaluate(env, disconnected, test_seeds)
                save_json(run / "evaluation.json", item)
                results["models"].append(item)
                save_json(output / "results.json", results)
                print(
                    f"{kind} seed={seed} val={metrics['best_val_mse']:.5f} "
                    f"clean={item['clean']['success_rate']:.1%} "
                    f"noise={item['noisy']['success_rate']:.1%}",
                    flush=True,
                )
                if seed == config.seeds[0] and kind == "connectome":
                    record(
                        env,
                        policy,
                        test_seeds[0],
                        output / "connectome-rollout.mp4",
                        episodes=min(6, len(test_seeds)),
                    )
        results.update(status="complete", elapsed_seconds=time.monotonic() - started)
        save_json(output / "results.json", results)
        return results
    finally:
        env.close()
