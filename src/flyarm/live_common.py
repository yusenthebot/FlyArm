"""Shared pieces of the live UI servers: session-guarded API, runtime loop and payloads.

`flyarm whole-brain serve` and `flyarm flyleg serve` build on these.
"""

from __future__ import annotations

import asyncio
import base64
import secrets
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import mujoco
import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from flyarm.pick_place_env import PandaPickPlaceEnv, physical_stage

CausalMode = Literal[
    "connectome", "shuffled", "edges_off", "direct_only", "deafferented", "head_deprived"
]


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


ACTIVITY_FLOOR = 1e-4
ROBOT_BODIES = (*(f"link{index}" for index in range(8)), "hand", "left_finger", "right_finger")


def _b64(array: np.ndarray, dtype: str) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode("ascii")


def encode_activity(values: np.ndarray) -> str:
    """Signed log-scale int8, base64: |state| in [1e-4, 1] maps to codes 1..127, below to 0.

    Rate states span four decades (saturated inputs near 1, most internal neurons near
    1e-3), so a linear int8 code would erase almost the whole brain. The log code keeps
    about 3.7% relative precision over the full range; the UI decodes it back to values.
    """
    state = np.clip(np.asarray(values, dtype=np.float64), -1.0, 1.0)
    magnitude = np.abs(state)
    decades = -np.log10(ACTIVITY_FLOOR)
    scaled = np.log10(np.maximum(magnitude, ACTIVITY_FLOOR) / ACTIVITY_FLOOR) / decades
    code = np.where(magnitude >= ACTIVITY_FLOOR, 1 + np.round(scaled * 126), 0)
    return _b64(np.sign(state) * code, "i1")


def decode_activity(encoded: str) -> np.ndarray:
    """Inverse of encode_activity (used by tests; the UI implements the same formula)."""
    code = np.frombuffer(base64.b64decode(encoded), dtype=np.int8).astype(np.float64)
    magnitude = ACTIVITY_FLOOR * 10 ** ((np.abs(code) - 1) / 126 * -np.log10(ACTIVITY_FLOOR))
    return np.where(code == 0, 0.0, np.sign(code) * magnitude)


def _robot_body_ids(model: mujoco.MjModel) -> list[int]:
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in ROBOT_BODIES]
    missing = [name for name, body in zip(ROBOT_BODIES, ids, strict=True) if body < 0]
    if missing:
        raise ValueError(f"Compiled model lacks Panda bodies: {missing}")
    return ids


def robot_body_poses(model: mujoco.MjModel, data: mujoco.MjData) -> list[list[float]]:
    """Live world pose [x, y, z, qw, qx, qy, qz] of every Panda body, in ROBOT_BODIES order."""
    return body_poses(data, _robot_body_ids(model))


def body_poses(data: mujoco.MjData, body_ids: list[int]) -> list[list[float]]:
    """Live world pose [x, y, z, qw, qx, qy, qz] of the given bodies, in order."""
    return np.round(np.hstack((data.xpos[body_ids], data.xquat[body_ids])), 6).tolist()


def _geom_color(model: mujoco.MjModel, geom: int) -> list[float]:
    """Material (or geom) RGBA; a textured material is tinted by its texture's mean color."""
    material = int(model.geom_matid[geom])
    if material < 0:
        return np.round(model.geom_rgba[geom], 4).tolist()
    rgba = model.mat_rgba[material].astype(np.float64).copy()
    texture_ids = np.atleast_1d(model.mat_texid[material])
    texture = next((int(t) for t in texture_ids if t >= 0), -1)
    if texture >= 0:
        start = int(model.tex_adr[texture])
        size = int(
            model.tex_width[texture] * model.tex_height[texture] * model.tex_nchannel[texture]
        )
        pixels = model.tex_data[start : start + size].reshape(-1, int(model.tex_nchannel[texture]))
        rgba[:3] *= pixels[:, :3].mean(axis=0) / 255.0
    return np.round(rgba, 4).tolist()


def build_scene_payload(
    model: mujoco.MjModel, body_ids: list[int], *, source: str
) -> dict[str, Any]:
    """Every visible geom of the given bodies exactly as MuJoCo compiled it, in body frames.

    Meshes come through the compiled model from its pinned assets, so the UI draws the same
    geometry MuJoCo simulates and renders (default visible groups 0-2). Mesh vertices are
    deduplicated by (position, normal); primitives carry MuJoCo's own size parameters.
    """
    body_index = {body: index for index, body in enumerate(body_ids)}
    geoms: list[dict[str, Any]] = []
    # Integer keys: MuJoCo >= 3.13 enums do not hash or compare equal to numpy integers.
    primitives = {
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
    }
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body not in body_index or model.geom_group[geom] > 2:
            continue
        rgba = _geom_color(model, geom)
        if rgba[3] <= 0:
            continue
        entry: dict[str, Any] = {
            "body": body_index[body],
            "pos": model.geom_pos[geom].tolist(),
            "quat": model.geom_quat[geom].tolist(),
            "rgba": rgba,
        }
        kind = int(model.geom_type[geom])
        if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh = int(model.geom_dataid[geom])
            v0 = model.mesh_vertadr[mesh]
            f0, fn = model.mesh_faceadr[mesh], model.mesh_facenum[mesh]
            faces = model.mesh_face[f0 : f0 + fn]
            corner_normals = model.mesh_normal[
                model.mesh_normaladr[mesh] + model.mesh_facenormal[f0 : f0 + fn]
            ]
            corners = np.concatenate(
                (model.mesh_vert[v0 + faces.reshape(-1)], corner_normals.reshape(-1, 3)), axis=1
            )
            unique, index = np.unique(np.round(corners, 5), axis=0, return_inverse=True)
            entry.update(
                kind="mesh",
                positions=_b64(unique[:, :3], "<f4"),
                normals=_b64(np.round(unique[:, 3:] * 127), "i1"),
                index=_b64(index.reshape(-1), "<u2" if len(unique) < 65536 else "<u4"),
                index_width=2 if len(unique) < 65536 else 4,
            )
        elif kind in primitives:
            entry.update(kind=primitives[kind], size=model.geom_size[geom].tolist())
        else:
            continue
        geoms.append(entry)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body) or "" for body in body_ids]
    return {
        "source": source,
        "bodies": names,
        "geoms": geoms,
        "pose_layout": "x y z qw qx qy qz, MuJoCo world frame (z up)",
        "size_convention": "MuJoCo geom_size: box half-extents, radius, half-length",
    }


def build_robot_payload(model: mujoco.MjModel) -> dict[str, Any]:
    """The pinned Menagerie Panda (plus FlyArm's rigid finger pads), body by body."""
    return build_scene_payload(
        model,
        _robot_body_ids(model),
        source="Google DeepMind MuJoCo Menagerie franka_emika_panda (Apache-2.0)",
    )


class LiveRuntime:
    """One MuJoCo episode advanced by an explicit selected policy."""

    def __init__(
        self,
        env: PandaPickPlaceEnv,
        controllers: Mapping[CausalMode, Any],
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
        if mode not in self.controllers:
            raise ValueError(f"Mode {mode} is not available for this run")
        with self._lock:
            self.mode = mode
            self.running = False
            self._observation, self._info = self.env.reset(
                seed=self._seed, options=self._reset_options
            )
            self._last_action = np.zeros(4, dtype=np.float32)
            for controller in self.controllers.values():
                controller.reset()

    @staticmethod
    def _hidden(controller: Any) -> np.ndarray:
        return controller.state[0].detach().cpu().numpy()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._compute_snapshot()

    def _compute_snapshot(self) -> dict[str, Any]:
        controller = self.controllers[self.mode]
        hidden = self._hidden(controller)
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
            "robot_bodies": robot_body_poses(self.env.model, self.env.data),
            "hidden_q": encode_activity(hidden),
            "hidden_count": int(hidden.size),
            "hidden_floor": ACTIVITY_FLOOR,
            "evidence": self.evidence,
        }

    @staticmethod
    def _physical_stage(info: Mapping[str, Any]) -> str:
        return physical_stage(dict(info))


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


def create_app(
    runtime: LiveRuntime,
    graph_payload: dict[str, Any],
    ui_dist: Path,
    neuron_details: Callable[[int], dict[str, Any]] | None = None,
    robot_payload: dict[str, Any] | None = None,
) -> FastAPI:
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

    @app.get("/api/robot")
    async def robot(request: Request) -> dict[str, Any]:
        require_session(request)
        if robot_payload is None:
            raise HTTPException(status_code=404, detail="Robot meshes are not served")
        return robot_payload

    @app.get("/api/neuron/{body_id}")
    async def neuron(body_id: int, request: Request) -> dict[str, Any]:
        require_session(request)
        if neuron_details is None:
            raise HTTPException(
                status_code=404, detail="Neuron details are not served for this run"
            )
        try:
            return neuron_details(body_id)
        except KeyError as error:
            raise HTTPException(status_code=404, detail=f"Unknown Body ID {body_id}") from error

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
        try:
            runtime.set_mode(body.mode)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
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
