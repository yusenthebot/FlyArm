"""Live UI for B1a checkpoints: the complete connectome drives the Panda in real time.

MLX binds GPU streams to threads, so one worker thread owns every MLX operation (acting,
resetting, reading state). HTTP handlers only enqueue commands and read the snapshot the
worker publishes after each change. The browser receives every neuron's state each frame
(int8, base64) and draws the neurons that have a measured soma location.
"""

from __future__ import annotations

import json
import queue
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import numpy as np
import pyarrow.feather as feather

from flyarm.interfaces import NeuralInterface
from flyarm.live import (
    CausalMode,
    LiveRuntime,
    ResetRequest,
    _b64,
    build_robot_payload,
    create_app,
)
from flyarm.pick_place_env import PandaPickPlaceEnv
from flyarm.whole_brain.backend_mlx import RateDynamics
from flyarm.whole_brain.compiler import ConnectomePack
from flyarm.whole_brain.diagnostics import direct_only_weights
from flyarm.whole_brain.experiment import load_trained_policy
from flyarm.whole_brain.policy import BrainPolicy, MlxController

CONTEXT_EDGES = 3000
PARTNERS = 12


class WholeBrainRuntime(LiveRuntime):
    """LiveRuntime whose controllers run on a dedicated MLX worker thread."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._commands: queue.Queue[tuple[Callable[[], None], threading.Event, list[Exception]]] = (
            queue.Queue()
        )
        self._published: dict[str, Any] = {}
        self._published_lock = threading.Lock()

    @staticmethod
    def _hidden(controller: Any) -> np.ndarray:
        return np.asarray(controller.state)[:, 0]

    def _submit(self, operation: Callable[[], None]) -> None:
        if self._thread is None:
            raise RuntimeError("Runtime worker is not running")
        done, error = threading.Event(), cast(list[Exception], [])
        self._commands.put((operation, done, error))
        if not done.wait(timeout=10.0):
            raise TimeoutError("Runtime worker did not respond")
        if error:
            raise error[0]

    def _publish(self) -> None:
        snapshot = self._compute_snapshot()
        with self._published_lock:
            self._published = snapshot

    def _loop(self) -> None:
        period = 1.0 / self.env.metadata["render_fps"]
        with mx.stream(mx.new_stream(mx.gpu)):
            for controller in self.controllers.values():
                controller.reset()
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
                        # Publish before signalling, so a caller never reads a stale state.
                        self._publish()
                        done.set()
                with self._lock:
                    if self.running:
                        self._advance()
                        self._publish()
                self._shutdown.wait(max(period - (time.monotonic() - started), 0.001))

    def run(self) -> None:
        self._submit(super().run)

    def pause(self) -> None:
        self._submit(super().pause)

    def step_once(self) -> None:
        self._submit(super().step_once)

    def reset(self, request: ResetRequest | None = None) -> None:
        self._submit(lambda: LiveRuntime.reset(self, request))

    def set_mode(self, mode: CausalMode) -> None:
        if mode not in self.controllers:
            raise ValueError(f"Mode {mode} is not available for this run")
        self._submit(lambda: LiveRuntime.set_mode(self, mode))

    def snapshot(self) -> dict[str, Any]:
        with self._published_lock:
            return dict(self._published)


class NeuronDetails:
    """Per-neuron annotation, degree and strongest measured partners, served on demand."""

    def __init__(self, pack: ConnectomePack, interface: NeuralInterface, drawn: np.ndarray) -> None:
        self.pack = pack
        self.neurons = pack.neurons().set_index("bodyId")
        inputs, outputs = interface.resolve_indices(pack)
        self.role = np.zeros(pack.nodes, dtype=np.int8)
        self.role[inputs], self.role[outputs] = 1, 2
        self.in_degree, self.out_degree = pack.degrees()
        self.rows = pack.rows()
        self.by_source = np.argsort(pack.col_idx, kind="stable")
        self.source_ptr = np.concatenate(([0], np.cumsum(self.out_degree)))
        self.drawn = drawn

    def _partners(self, indices: np.ndarray, contacts: np.ndarray) -> list[dict[str, Any]]:
        order = np.argsort(-contacts, kind="stable")[:PARTNERS]
        return [
            {
                "id": int(self.pack.body_ids[indices[k]]),
                "contacts": int(contacts[k]),
                "drawn": bool(self.drawn[indices[k]]),
                "type": self._text(int(self.pack.body_ids[indices[k]]), "type"),
            }
            for k in order
        ]

    def _text(self, body_id: int, column: str) -> str | None:
        value = self.neurons.at[body_id, column]
        return value if isinstance(value, str) else None

    def __call__(self, body_id: int) -> dict[str, Any]:
        index = int(np.searchsorted(self.pack.body_ids, body_id))
        if index >= self.pack.nodes or int(self.pack.body_ids[index]) != body_id:
            raise KeyError(body_id)
        start, stop = self.pack.row_ptr[index], self.pack.row_ptr[index + 1]
        upstream = self._partners(
            np.asarray(self.pack.col_idx[start:stop]), np.asarray(self.pack.contacts[start:stop])
        )
        slots = self.by_source[self.source_ptr[index] : self.source_ptr[index + 1]]
        downstream = self._partners(self.rows[slots], np.asarray(self.pack.contacts[slots]))
        return {
            "id": body_id,
            "index": index,
            "type": self._text(body_id, "type"),
            "superclass": self._text(body_id, "superclass"),
            "class": self._text(body_id, "class"),
            "consensus_nt": self._text(body_id, "consensus_nt"),
            "sign": int(self.pack.signs[index]),
            "role": ("internal", "input", "output")[self.role[index]],
            "in_degree": int(self.in_degree[index]),
            "out_degree": int(self.out_degree[index]),
            "upstream": upstream,
            "downstream": downstream,
        }


def build_whole_brain_payload(
    pack: ConnectomePack,
    interface: NeuralInterface,
    annotations_path: Path,
    modes: Sequence[str],
) -> tuple[dict[str, Any], np.ndarray]:
    """Columnar payload: every neuron with a measured soma, plus context edges."""
    table = feather.read_table(annotations_path, columns=["bodyId", "somaLocation"]).to_pandas()
    soma = table.set_index("bodyId").somaLocation.reindex(pack.body_ids)
    drawn = soma.notna().to_numpy()
    xyz = np.stack(soma[drawn].to_numpy()).astype(np.float64)
    center = (xyz.min(axis=0) + xyz.max(axis=0)) / 2
    scale = float(np.max(np.ptp(xyz, axis=0)))
    # Normalized and Y-up with MaleCNS Z vertical as in the subgraph view, then turned 180
    # degrees about the view axis (a rotation, not a mirror) so the brain sits above the VNC.
    positions = ((xyz - center) / scale * 1.9)[:, [0, 2, 1]] * np.array([-1.0, -1.0, 1.0])
    state_index = np.flatnonzero(drawn)
    inputs, outputs = interface.resolve_indices(pack)
    role = np.zeros(pack.nodes, dtype=np.uint8)
    role[inputs], role[outputs] = 1, 2
    drawn_slot = np.full(pack.nodes, -1, dtype=np.int64)
    drawn_slot[state_index] = np.arange(len(state_index))

    rows = pack.rows()
    direct = (role[pack.col_idx] == 1) & (role[rows] == 2)
    direct &= (drawn_slot[pack.col_idx] >= 0) & (drawn_slot[rows] >= 0)
    candidates = np.flatnonzero(direct)
    strongest = candidates[np.argsort(-pack.contacts[candidates], kind="stable")[:CONTEXT_EDGES]]
    edge_pairs = np.stack((drawn_slot[pack.col_idx[strongest]], drawn_slot[rows[strongest]]), 1)

    neurons = pack.neurons()
    undrawn = neurons.superclass[~drawn].value_counts().to_dict()
    payload = {
        "format": "columnar-v1",
        "dataset": pack.manifest["dataset"],
        "layout": "measured_soma_locations",
        "title": f"complete CNS · {pack.nodes:,} neurons · {pack.edges:,} edges",
        "graph_fingerprint": pack.fingerprint(),
        "interface_fingerprint": interface.fingerprint,
        "neurons": pack.nodes,
        "drawn": int(drawn.sum()),
        "undrawn_by_superclass": {str(k): int(v) for k, v in undrawn.items()},
        "ids": _b64(pack.body_ids[state_index], "<u4"),
        "state_index": _b64(state_index, "<u4"),
        "positions": _b64(positions, "<f4"),
        "roles": _b64(role[state_index], "u1"),
        "context_edges": _b64(edge_pairs, "<u4"),
        "context_edge_note": (
            f"strongest {len(strongest):,} of {int(direct.sum()):,} drawn direct "
            "ascending->output synapses"
        ),
        "modes": list(modes),
    }
    if payload["drawn"] >= 2**32:
        raise ValueError("Too many neurons for uint32 indices")
    return payload, drawn


def summarize_whole_brain_evidence(results: dict[str, Any] | None) -> dict[str, Any]:
    """B1a reading rules; nothing is promoted without a complete three-seed run."""
    summary: dict[str, Any] = {
        "graph_mediated": "pending",
        "topology_advantage": "pending",
        "connectome_success_rate": None,
        "shuffled_success_rate": None,
        "edges_off_success_rate": None,
        "direct_only_success_rate": None,
        "seeds": 0,
    }
    if not results or results.get("status") != "complete":
        return summary
    evidence = results.get("evidence", {})
    brain = evidence.get("connectome_success")
    shuffled = evidence.get("shuffled_success")
    edges_off = evidence.get("edges_off_success")
    seeds = int(evidence.get("connectome_seeds", 0))
    summary.update(
        connectome_success_rate=brain,
        shuffled_success_rate=shuffled,
        edges_off_success_rate=edges_off,
        direct_only_success_rate=evidence.get("direct_only_success"),
        seeds=seeds,
    )
    if brain is not None and edges_off is not None:
        if brain > 0.7 and edges_off < 0.1:
            summary["graph_mediated"] = "supported" if seeds >= 3 else "pending"
        else:
            summary["graph_mediated"] = "not_supported"
    if brain is not None and shuffled is not None:
        if brain < 0.7 or brain - shuffled < 0.2:
            summary["topology_advantage"] = "not_supported"
    return summary


def serve_whole_brain(
    run_root: Path,
    pack_root: Path,
    annotations_path: Path,
    model_path: Path,
    ui_dist: Path,
    *,
    host: str,
    port: int,
    seed: int,
    episode: int,
) -> None:
    import uvicorn

    from flyarm.live import _validate_server_scope

    ui_dist = _validate_server_scope(ui_dist, host)
    config = json.loads((run_root / "config.json").read_text())
    if config.get("task") != "pick-place":
        raise ValueError("The live UI drives the pick-and-place task")
    task, policy = load_trained_policy(run_root, "connectome", seed, pack_root, model_path)
    if not isinstance(policy, BrainPolicy) or not isinstance(task.env, PandaPickPlaceEnv):
        raise ValueError("Expected a whole-brain pick-and-place checkpoint")
    pack = ConnectomePack.load(pack_root)
    interface = NeuralInterface.load(run_root / "interface.json")
    controllers: dict[CausalMode, MlxController] = {
        "connectome": MlxController(policy),
        "edges_off": MlxController(
            policy.with_dynamics("edges_off", RateDynamics(pack, interface, edges=False))
        ),
        "direct_only": MlxController(
            policy.with_dynamics(
                "direct_only",
                RateDynamics(pack, interface, weights=direct_only_weights(pack, interface)),
            )
        ),
    }
    if (run_root / f"shuffled-{seed}" / "policy.safetensors").is_file():
        shuffled_task, shuffled = load_trained_policy(
            run_root, "shuffled", seed, pack_root, model_path
        )
        shuffled_task.close()
        controllers["shuffled"] = MlxController(shuffled)
    for controller in controllers.values():
        mx.eval(controller.policy.parameters())
    modes = [
        mode
        for mode in ("connectome", "shuffled", "edges_off", "direct_only")
        if mode in controllers
    ]
    payload, drawn = build_whole_brain_payload(pack, interface, annotations_path, modes)
    results_path = run_root / "results.json"
    results = json.loads(results_path.read_text()) if results_path.is_file() else None
    runtime = WholeBrainRuntime(
        task.env, controllers, summarize_whole_brain_evidence(results), seed=episode
    )
    runtime.start()
    details = NeuronDetails(pack, interface, drawn)
    uvicorn.run(
        create_app(
            runtime,
            payload,
            ui_dist,
            neuron_details=details,
            robot_payload=build_robot_payload(task.env.model),
        ),
        host=host,
        port=port,
        log_level="info",
    )
