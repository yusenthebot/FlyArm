import type { CausalMode, GraphPayload, NeuronDetail, RobotPayload, RuntimeState } from "./types";

const sessionToken =
  document.querySelector<HTMLMetaElement>('meta[name="flyarm-session"]')?.content ?? "";

if (!sessionToken || sessionToken === "__FLYARM_SESSION__") {
  throw new Error("FlyArm UI must be loaded through the authenticated live server");
}

async function post(path: string, body: object = {}): Promise<void> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-FlyArm-Session": sessionToken },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status} ${await response.text()}`);
  }
}

export async function loadGraph(): Promise<GraphPayload> {
  const response = await fetch("/api/graph", {
    headers: { "X-FlyArm-Session": sessionToken },
  });
  if (!response.ok) throw new Error(`Graph load failed: ${response.status}`);
  return response.json() as Promise<GraphPayload>;
}

/** Resolves to null on servers that predate the mesh endpoint (skeleton fallback). */
export async function loadRobot(): Promise<RobotPayload | null> {
  const response = await fetch("/api/robot", {
    headers: { "X-FlyArm-Session": sessionToken },
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`Robot model load failed: ${response.status}`);
  return response.json() as Promise<RobotPayload>;
}

export async function loadNeuron(id: number): Promise<NeuronDetail> {
  const response = await fetch(`/api/neuron/${id}`, {
    headers: { "X-FlyArm-Session": sessionToken },
  });
  if (!response.ok) throw new Error(`Neuron ${id} failed: ${response.status}`);
  return response.json() as Promise<NeuronDetail>;
}

export const controls = {
  reset: (object?: [number, number, number], goal?: [number, number, number]) =>
    post("/api/reset", { object, goal }),
  run: () => post("/api/run"),
  pause: () => post("/api/pause"),
  step: () => post("/api/step"),
  mode: (mode: CausalMode) => post("/api/mode", { mode }),
};

export function openStateStream(
  onState: (state: RuntimeState) => void,
  onConnection: (connected: boolean) => void,
): () => void {
  let closed = false;
  let socket: WebSocket | null = null;
  let retry: number | undefined;

  const connect = () => {
    if (closed) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    socket = new WebSocket(`${protocol}://${window.location.host}/ws/state`, [
      "flyarm",
      sessionToken,
    ]);
    socket.onopen = () => onConnection(true);
    socket.onmessage = (event) => onState(JSON.parse(event.data) as RuntimeState);
    socket.onclose = () => {
      onConnection(false);
      if (!closed) retry = window.setTimeout(connect, 1000);
    };
    socket.onerror = () => socket?.close();
  };
  connect();
  return () => {
    closed = true;
    if (retry !== undefined) window.clearTimeout(retry);
    socket?.close();
  };
}
