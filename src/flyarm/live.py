"""Real-time, auditable FlyArm simulation server for the 256-neuron subgraph runs."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.feather as feather
import torch

from flyarm.config import PickPlaceConfig
from flyarm.graph import Graph
from flyarm.interfaces import NeuralInterface, canonical_interface
from flyarm.live_common import (
    CausalMode,
    LiveRuntime,
    _validate_server_scope,
    build_robot_payload,
    create_app,
)
from flyarm.pick_place_env import PandaPickPlaceEnv
from flyarm.pick_place_models import PickPlaceController, PickPlacePolicy


def build_graph_payload(
    graph: Graph, interface: NeuralInterface, annotations_path: Path
) -> dict[str, Any]:
    """Bind measured soma coordinates and annotations to exact graph IDs."""
    interface.resolve_indices(graph)
    if not annotations_path.is_file():
        raise FileNotFoundError(annotations_path)
    table = feather.read_table(
        annotations_path,
        columns=["bodyId", "type", "superclass", "somaLocation"],
    ).to_pandas()
    table = table[table.bodyId.isin(graph.ids)].set_index("bodyId")
    missing = sorted(set(map(int, graph.ids)) - set(map(int, table.index)))
    if missing:
        raise ValueError(f"Annotations missing graph neuron IDs: {missing[:5]}")

    measured = [
        np.asarray(table.at[int(body_id), "somaLocation"], dtype=np.float64)
        if table.at[int(body_id), "somaLocation"] is not None
        else None
        for body_id in graph.ids
    ]
    available = np.stack([position for position in measured if position is not None])
    center = (available.min(axis=0) + available.max(axis=0)) / 2
    scale = float(np.max(np.ptp(available, axis=0)))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("MaleCNS soma coordinates have invalid extent")
    positions = []
    for position in measured:
        normalized = ((position if position is not None else center) - center) / scale * 1.9
        # Three.js is Y-up; preserve measured coordinates while placing MaleCNS Z vertically.
        positions.append(normalized[[0, 2, 1]])
    inputs, outputs = (
        set(map(int, interface.input_body_ids)),
        set(map(int, interface.output_body_ids)),
    )
    in_degree = np.bincount(graph.post, minlength=len(graph.ids))
    out_degree = np.bincount(graph.pre, minlength=len(graph.ids))
    nodes: list[dict[str, Any]] = []
    for index, (body_id, position, measured_position) in enumerate(
        zip(graph.ids, positions, measured, strict=True)
    ):
        integer_id = int(body_id)
        role = (
            "input" if integer_id in inputs else "output" if integer_id in outputs else "internal"
        )
        row = table.loc[integer_id]
        neuron_type = row["type"] if isinstance(row["type"], str) else None
        superclass = row["superclass"] if isinstance(row["superclass"], str) else None
        nodes.append(
            {
                "id": integer_id,
                "index": index,
                "position": position.tolist(),
                "soma_location": (
                    measured_position.tolist() if measured_position is not None else None
                ),
                "type": neuron_type,
                "superclass": superclass,
                "role": role,
                "sign": int(graph.signs[index]),
                "in_degree": int(in_degree[index]),
                "out_degree": int(out_degree[index]),
            }
        )
    normalized = graph.normalized_weights()
    edges = [
        {
            "source": int(graph.ids[source]),
            "target": int(graph.ids[target]),
            "contacts": float(contacts),
            "weight": float(weight),
        }
        for source, target, contacts, weight in zip(
            graph.pre, graph.post, graph.contacts, normalized, strict=True
        )
    ]
    return {
        "nodes": nodes,
        "edges": edges,
        "graph_fingerprint": graph.fingerprint(),
        "interface_fingerprint": interface.fingerprint,
        "dataset": graph.metadata["dataset"],
        "layout": "measured_soma_locations",
        "format": "nodes-v1",
        "title": f"{len(graph.ids)} measured neurons · {len(graph.pre):,} synaptic edges",
        "modes": ["connectome", "shuffled", "edges_off"],
    }


def summarize_evidence(results: Mapping[str, Any] | None) -> dict[str, Any]:
    """Conservative UI summary; absence or failed experiments remain pending."""
    empty = {
        "graph_mediated": "pending",
        "topology_advantage": "pending",
        "connectome_success_rate": None,
        "shuffled_success_rate": None,
        "edges_off_success_rate": None,
    }
    if not results or results.get("status") != "complete":
        return empty
    models = results.get("models")
    if not isinstance(models, list):
        return empty

    def rates(kind: str, field: str = "clean") -> dict[int, float]:
        values: dict[int, float] = {}
        for item in models:
            if not isinstance(item, Mapping) or item.get("kind") != kind:
                continue
            evaluation = item.get(field)
            seed = item.get("seed")
            if (
                isinstance(seed, int)
                and isinstance(evaluation, Mapping)
                and isinstance(evaluation.get("success_rate"), (int, float))
            ):
                values[seed] = float(evaluation["success_rate"])
        return values

    true_rates = rates("restricted_connectome")
    shuffle_rates = rates("restricted_shuffled")
    edges_off_rates = rates("restricted_connectome", "edges_silenced")
    true_rate = float(np.mean(list(true_rates.values()))) if true_rates else None
    shuffle_rate = float(np.mean(list(shuffle_rates.values()))) if shuffle_rates else None
    edges_off_rate = float(np.mean(list(edges_off_rates.values()))) if edges_off_rates else None
    graph_mediated = "pending"
    mediation_seeds = sorted(true_rates.keys() & edges_off_rates.keys())
    if mediation_seeds:
        matched_true = float(np.mean([true_rates[seed] for seed in mediation_seeds]))
        matched_edges_off = float(np.mean([edges_off_rates[seed] for seed in mediation_seeds]))
        if matched_true < 0.5 or matched_true - matched_edges_off < 0.5:
            graph_mediated = "not_supported"
        elif len(mediation_seeds) >= 3:
            graph_mediated = "supported"
    topology_advantage = "pending"
    comparison_seeds = sorted(true_rates.keys() & shuffle_rates.keys())
    if comparison_seeds:
        matched_true = float(np.mean([true_rates[seed] for seed in comparison_seeds]))
        matched_shuffle = float(np.mean([shuffle_rates[seed] for seed in comparison_seeds]))
        if matched_true < 0.5 or matched_true - matched_shuffle < 0.2:
            topology_advantage = "not_supported"
        elif len(comparison_seeds) >= 3:
            topology_advantage = "supported"
    return {
        "graph_mediated": graph_mediated,
        "topology_advantage": topology_advantage,
        "connectome_success_rate": true_rate,
        "shuffled_success_rate": shuffle_rate,
        "edges_off_success_rate": edges_off_rate,
    }


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in loaded.items()
    ):
        raise ValueError(f"Checkpoint is not a tensor state dict: {path}")
    return loaded


def load_live_controllers(
    graph: Graph,
    interface: NeuralInterface,
    run_path: Path,
    seed: int,
    internal_steps: int,
) -> dict[CausalMode, PickPlaceController]:
    true_state = _load_state(run_path / f"restricted_connectome-{seed}" / "policy.pt")
    obs_dim = int(true_state["obs_mean"].numel())
    true_policy = PickPlacePolicy(
        "restricted_connectome",
        graph,
        interface,
        seed=seed,
        obs_dim=obs_dim,
        internal_steps=internal_steps,
    )
    true_policy.load_state_dict(true_state)

    disconnected = PickPlacePolicy(
        "restricted_disconnected",
        graph,
        interface,
        seed=seed,
        obs_dim=obs_dim,
        internal_steps=internal_steps,
    )
    disconnected.load_state_dict(
        {key: value for key, value in true_state.items() if key != "adjacency"}, strict=False
    )

    shuffled_graph = Graph.load(run_path / f"shuffled-{seed}.npz")
    shuffled_interface = NeuralInterface.bind(
        shuffled_graph,
        interface.input_body_ids,
        interface.output_body_ids,
        label=interface.label,
    )
    shuffled = PickPlacePolicy(
        "restricted_shuffled",
        shuffled_graph,
        shuffled_interface,
        seed=seed,
        obs_dim=obs_dim,
        internal_steps=internal_steps,
    )
    shuffled.load_state_dict(_load_state(run_path / f"restricted_shuffled-{seed}" / "policy.pt"))
    return {
        "connectome": PickPlaceController(true_policy),
        "shuffled": PickPlaceController(shuffled),
        "edges_off": PickPlaceController(disconnected),
    }


def serve_live(
    graph_path: Path,
    annotations_path: Path,
    model_path: Path,
    run_path: Path,
    ui_dist: Path,
    *,
    host: str,
    port: int,
    seed: int,
) -> None:
    import uvicorn

    from flyarm.assets import verify_arm

    ui_dist = _validate_server_scope(ui_dist, host)
    verified_scene = verify_arm(model_path.resolve().parent.parent)
    if verified_scene.resolve() != model_path.resolve():
        raise ValueError("Model must be the pinned, unmodified Menagerie Panda scene.xml")
    graph = Graph.load(graph_path)
    graph.validate_mvp_provenance()
    interface = canonical_interface(graph)
    run_graph = Graph.load(run_path / "graph.npz")
    if run_graph.fingerprint() != graph.fingerprint():
        raise ValueError("Run graph fingerprint does not match the requested graph")
    run_interface = NeuralInterface.load(run_path / "interface.json")
    if run_interface != interface:
        raise ValueError("Run neural interface does not match the canonical interface")
    run_config = PickPlaceConfig.model_validate_json((run_path / "config.json").read_text())
    if seed not in run_config.seeds:
        raise ValueError(f"Seed {seed} is not present in the saved run configuration")
    payload = build_graph_payload(graph, interface, annotations_path)
    payload["internal_steps"] = run_config.internal_steps
    results_path = run_path / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else None
    controllers = load_live_controllers(graph, interface, run_path, seed, run_config.internal_steps)
    runtime = LiveRuntime(
        PandaPickPlaceEnv(model_path=model_path, horizon=run_config.horizon),
        controllers,
        summarize_evidence(results),
        seed=seed,
    )
    runtime.start()
    app = create_app(
        runtime, payload, ui_dist, robot_payload=build_robot_payload(runtime.env.model)
    )
    uvicorn.run(app, host=host, port=port, log_level="info")
