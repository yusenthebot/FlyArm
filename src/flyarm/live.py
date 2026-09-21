"""Real-time, auditable FlyArm simulation server for the interactive UI."""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
import pyarrow.feather as feather
import torch
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from flyarm.config import PickPlaceConfig
from flyarm.graph import Graph
from flyarm.interfaces import NeuralInterface, canonical_interface
from flyarm.pick_place_env import PandaPickPlaceEnv
from flyarm.pick_place_models import PickPlaceController, PickPlacePolicy

CausalMode = Literal["connectome", "shuffled", "edges_off"]


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    object: tuple[float, float, float] | None = None
    goal: tuple[float, float, float] | None = None


class ModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: CausalMode


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


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


class LiveRuntime:
    """One MuJoCo episode advanced by an explicit selected policy."""

    def __init__(
        self,
        env: PandaPickPlaceEnv,
        controllers: dict[CausalMode, PickPlaceController],
        evidence: dict[str, Any],
        *,
        seed: int = 0,
    ) -> None:
        self.env = env
        self.controllers = controllers
        self.evidence = evidence
        self.mode: CausalMode = "connectome"
        self.running = False
        self._seed = seed
        self._lock = threading.RLock()
        self._shutdown = threading.Event()
        self._thread: threading.Thread | None = None
        self._observation, self._info = self.env.reset(seed=seed)
        self._last_action = np.zeros(4, dtype=np.float32)
        self._reset_options: dict[str, np.ndarray] | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="flyarm-live", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._shutdown.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.env.close()

    def _loop(self) -> None:
        period = 1.0 / self.env.metadata["render_fps"]
        while not self._shutdown.is_set():
            started = time.monotonic()
            with self._lock:
                if self.running:
                    self._advance()
            self._shutdown.wait(max(period - (time.monotonic() - started), 0.001))

    def _advance(self) -> None:
        controller = self.controllers[self.mode]
        self._last_action = controller.act(self._observation)
        self._observation, _, terminated, truncated, self._info = self.env.step(self._last_action)
        if terminated or truncated:
            self.running = False

    def run(self) -> None:
        with self._lock:
            self.running = True

    def pause(self) -> None:
        with self._lock:
            self.running = False

    def step_once(self) -> None:
        with self._lock:
            self.running = False
            self._advance()

    def reset(self, request: ResetRequest | None = None) -> None:
        with self._lock:
            self.running = False
            options: dict[str, np.ndarray] | None = None
            if request is not None and (request.object is not None or request.goal is not None):
                if request.object is None or request.goal is None:
                    raise ValueError("Interactive reset requires both object and goal coordinates")
                options = {
                    "object": self._workspace_vector(request.object, "object", 0.02),
                    "goal": self._workspace_vector(request.goal, "goal", 0.002),
                }
            self._reset_options = options
            self._observation, self._info = self.env.reset(seed=self._seed, options=options)
            self._last_action = np.zeros(4, dtype=np.float32)
            for controller in self.controllers.values():
                controller.reset()

    @staticmethod
    def _workspace_vector(
        value: tuple[float, float, float], name: str, expected_z: float
    ) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if (
            not np.all(np.isfinite(vector))
            or not 0.20 <= vector[0] <= 0.70
            or not -0.30 <= vector[1] <= 0.30
            or abs(vector[2] - expected_z) > 0.025
        ):
            raise ValueError(f"{name} position is outside the bounded tabletop workspace")
        vector[2] = expected_z
        return vector

    def set_mode(self, mode: CausalMode) -> None:
        with self._lock:
            self.mode = mode
            self.running = False
            self._observation, self._info = self.env.reset(
                seed=self._seed, options=self._reset_options
            )
            self._last_action = np.zeros(4, dtype=np.float32)
            for controller in self.controllers.values():
                controller.reset()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            controller = self.controllers[self.mode]
            hidden = controller.state[0].detach().cpu().numpy()
            robot_points: list[list[float]] = []
            for name in [*(f"link{index}" for index in range(8)), "hand"]:
                body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    robot_points.append(self.env.data.xpos[body_id].tolist())
            gripper_points = []
            for name in ["left_finger", "right_finger"]:
                body_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id >= 0:
                    gripper_points.append(self.env.data.xpos[body_id].tolist())
            info = {key: _json_value(value) for key, value in self._info.items()}
            return {
                "connected": True,
                "running": self.running,
                "mode": self.mode,
                "step": self.env.steps,
                "sim_time": float(self.env.data.time),
                "stage": self._physical_stage(info),
                "success": bool(info["is_success"]),
                "contact_left": bool(info["contact_left"]),
                "contact_right": bool(info["contact_right"]),
                "ever_grasped": bool(info["ever_grasped"]),
                "ever_lifted": bool(info["ever_lifted"]),
                "object_height": float(info["object_height"]),
                "goal_error": float(info["goal_xy_error"]),
                "gripper_opening": float(info["gripper_opening"]),
                "object": info["object_position"],
                "goal": info["goal"],
                "end_effector": info["ee_position"],
                "robot_points": robot_points,
                "gripper_points": gripper_points,
                "action": self._last_action.tolist(),
                "hidden": hidden.tolist(),
                "evidence": self.evidence,
            }

    @staticmethod
    def _physical_stage(info: Mapping[str, Any]) -> str:
        if bool(info["is_success"]):
            return "placed"
        if bool(info["ever_lifted"]):
            return "lifted"
        if bool(info["ever_grasped"]):
            return "grasped"
        if bool(info["contact_left"]) or bool(info["contact_right"]):
            return "contact"
        return "free"


def _approved_ui_dist() -> Path:
    return Path(__file__).resolve().parents[2] / "ui" / "dist"


def _validate_server_scope(ui_dist: Path, host: str) -> Path:
    if host not in {"127.0.0.1", "::1", "localhost"}:
        raise ValueError("Live UI only supports loopback hosts")
    resolved = ui_dist.resolve()
    approved = _approved_ui_dist().resolve()
    if resolved != approved:
        raise ValueError(f"UI directory must be the project build at {approved}")
    return resolved


def create_app(runtime: LiveRuntime, graph_payload: dict[str, Any], ui_dist: Path) -> FastAPI:
    if not (ui_dist / "index.html").is_file():
        raise FileNotFoundError(f"Built UI not found: {ui_dist}; run `npm --prefix ui run build`")

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runtime.close()

    app = FastAPI(title="FlyArm Live", docs_url=None, redoc_url=None, lifespan=lifespan)
    session_token = secrets.token_urlsafe(32)
    session_cookie = "flyarm_live_session"
    websocket_slots = asyncio.Semaphore(4)

    def require_session(request: Request) -> None:
        supplied = request.headers.get("x-flyarm-session") or request.cookies.get(
            session_cookie, ""
        )
        if not secrets.compare_digest(supplied, session_token):
            raise HTTPException(status_code=401, detail="Open the FlyArm UI before using its API")
        origin = request.headers.get("origin")
        expected = f"{request.url.scheme}://{request.headers.get('host', '')}"
        if origin is not None and origin != expected:
            raise HTTPException(status_code=403, detail="Cross-origin control is not allowed")

    @app.get("/", include_in_schema=False)
    async def index() -> HTMLResponse:
        document = (ui_dist / "index.html").read_text()
        marker = "__FLYARM_SESSION__"
        if marker not in document:
            raise HTTPException(status_code=500, detail="UI build is missing the session marker")
        response = HTMLResponse(document.replace(marker, session_token))
        response.set_cookie(
            session_cookie,
            session_token,
            httponly=True,
            samesite="strict",
            secure=False,
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/api/graph")
    async def graph(request: Request) -> dict[str, Any]:
        require_session(request)
        return graph_payload

    @app.get("/api/state")
    async def state(request: Request) -> dict[str, Any]:
        require_session(request)
        return runtime.snapshot()

    @app.post("/api/run")
    async def run(request: Request) -> dict[str, bool]:
        require_session(request)
        runtime.run()
        return {"ok": True}

    @app.post("/api/pause")
    async def pause(request: Request) -> dict[str, bool]:
        require_session(request)
        runtime.pause()
        return {"ok": True}

    @app.post("/api/step")
    async def step(request: Request) -> dict[str, bool]:
        require_session(request)
        runtime.step_once()
        return {"ok": True}

    @app.post("/api/reset")
    async def reset(body: ResetRequest, request: Request) -> dict[str, bool]:
        require_session(request)
        try:
            runtime.reset(body)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {"ok": True}

    @app.post("/api/mode")
    async def mode(body: ModeRequest, request: Request) -> dict[str, bool]:
        require_session(request)
        runtime.set_mode(body.mode)
        return {"ok": True}

    @app.websocket("/ws/state")
    async def websocket_state(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        host_header = websocket.headers.get("host", "")
        expected_origin = f"http://{host_header}"
        protocols = [
            item.strip() for item in websocket.headers.get("sec-websocket-protocol", "").split(",")
        ]
        supplied = protocols[1] if len(protocols) == 2 and protocols[0] == "flyarm" else ""
        if origin != expected_origin or not secrets.compare_digest(supplied, session_token):
            await websocket.close(code=1008)
            return
        try:
            await asyncio.wait_for(websocket_slots.acquire(), timeout=0.1)
        except TimeoutError:
            await websocket.close(code=1013)
            return
        await websocket.accept(subprotocol="flyarm")
        try:
            while True:
                await asyncio.wait_for(websocket.send_json(runtime.snapshot()), timeout=0.2)
                await asyncio.sleep(0.05)
        except (TimeoutError, WebSocketDisconnect):
            return
        finally:
            websocket_slots.release()

    app.mount("/", StaticFiles(directory=ui_dist, html=True), name="ui")
    return app


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
    uvicorn.run(create_app(runtime, payload, ui_dist), host=host, port=port, log_level="info")
