export type CausalMode =
  | "connectome"
  | "shuffled"
  | "edges_off"
  | "direct_only"
  | "deafferented"
  | "head_deprived";

export interface GraphNode {
  id: number;
  index: number;
  position: [number, number, number];
  soma_location: [number, number, number] | null;
  type: string | null;
  superclass: string | null;
  role: "input" | "internal" | "output";
  sign: -1 | 0 | 1;
  in_degree: number;
  out_degree: number;
}

export interface GraphEdge {
  source: number;
  target: number;
  contacts: number;
  weight: number;
}

interface PayloadCommon {
  graph_fingerprint: string;
  interface_fingerprint: string;
  dataset: string;
  layout: "measured_soma_locations";
  title?: string;
  modes?: CausalMode[];
}

export interface SubgraphPayload extends PayloadCommon {
  format?: "nodes-v1";
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/** Complete connectome: binary columns (base64) for every neuron with a measured soma. */
export interface WholeBrainPayload extends PayloadCommon {
  format: "columnar-v1";
  task?: "pick-place" | "kitchen";
  role_labels?: Record<string, string>;
  proxy_positioned_afferents?: number;
  neurons: number;
  drawn: number;
  undrawn_by_superclass: Record<string, number>;
  ids: string;
  state_index: string;
  positions: string;
  roles: string;
  context_edges: string;
  context_edge_note: string;
}

export type GraphPayload = SubgraphPayload | WholeBrainPayload;

interface RobotGeomBase {
  body: number;
  pos: [number, number, number];
  quat: [number, number, number, number];
  rgba: [number, number, number, number];
}

/** Primitive sizes are MuJoCo geom_size: box half-extents, radius, half-length. */
export type RobotGeom =
  | (RobotGeomBase & { kind: "mesh"; positions: string; normals: string; index: string; index_width: 2 | 4 })
  | (RobotGeomBase & { kind: "box" | "sphere" | "cylinder" | "capsule" | "plane"; size: [number, number, number] });

/** Visible Panda geoms from the compiled Menagerie model, in body frames. */
export interface RobotPayload {
  source: string;
  bodies: string[];
  geoms: RobotGeom[];
}

export interface Partner {
  id: number;
  contacts: number;
  drawn: boolean;
  type: string | null;
}

export interface NeuronDetail {
  id: number;
  index: number;
  type: string | null;
  superclass: string | null;
  class: string | null;
  consensus_nt: string | null;
  sign: -1 | 0 | 1;
  role: "input" | "internal" | "output";
  in_degree: number;
  out_degree: number;
  upstream: Partner[];
  downstream: Partner[];
}

export interface EvidenceSummary {
  graph_mediated: "supported" | "not_supported" | "pending";
  topology_advantage: "supported" | "not_supported" | "pending";
  connectome_success_rate: number | null;
  shuffled_success_rate: number | null;
  edges_off_success_rate: number | null;
  direct_only_success_rate?: number | null;
  seeds?: number;
}

export interface RuntimeState {
  connected: boolean;
  running: boolean;
  mode: CausalMode;
  step: number;
  sim_time: number;
  stage: string;
  success: boolean;
  contact_left: boolean;
  contact_right: boolean;
  ever_grasped: boolean;
  ever_lifted: boolean;
  object_height: number;
  goal_error: number;
  gripper_opening: number;
  object: [number, number, number];
  goal: [number, number, number];
  end_effector: [number, number, number];
  robot_points: [number, number, number][];
  /** Per body in RobotPayload order: [x, y, z, qw, qx, qy, qz] (MuJoCo world frame). */
  robot_bodies?: number[][];
  gripper_points: [number, number, number][];
  action: [number, number, number, number];
  /** int8 state, base64, value / 127; older servers send `hidden` as floats. */
  hidden_q?: string;
  hidden_count?: number;
  hidden_floor?: number;
  /** FrankaKitchen only. */
  tasks?: string[];
  completed?: string[];
  task_distance?: Record<string, number>;
  score?: number;
  /** Chunked controllers: newest predicted chunk, [k][9] joint velocities, control_dt apart. */
  plan?: number[][] | null;
  control_dt?: number;
  hidden?: number[];
  evidence: EvidenceSummary;
}

export interface TimelineSample {
  step: number;
  time: number;
  action: [number, number, number, number];
  objectHeight: number;
  goalError: number;
  activation: number;
  left: boolean;
  right: boolean;
}

export const EMPTY_STATE: RuntimeState = {
  connected: false,
  running: false,
  mode: "connectome",
  step: 0,
  sim_time: 0,
  stage: "waiting",
  success: false,
  contact_left: false,
  contact_right: false,
  ever_grasped: false,
  ever_lifted: false,
  object_height: 0,
  goal_error: 0,
  gripper_opening: 1,
  object: [0, 0, 0.02],
  goal: [0.1, 0, 0.002],
  end_effector: [0, 0, 0],
  robot_points: [],
  gripper_points: [],
  action: [0, 0, 0, 1],
  evidence: {
    graph_mediated: "pending",
    topology_advantage: "pending",
    connectome_success_rate: null,
    shuffled_success_rate: null,
    edges_off_success_rate: null,
  },
};

export function decodeActivity(state: RuntimeState): Float32Array {
  if (state.hidden_q) {
    const bytes = Uint8Array.from(atob(state.hidden_q), (char) => char.charCodeAt(0));
    const signed = new Int8Array(bytes.buffer);
    const values = new Float32Array(signed.length);
    // Signed log code (see flyarm.live_common.encode_activity): codes 1..127 span floor..1.
    const floor = state.hidden_floor ?? 1e-4;
    const decades = -Math.log10(floor);
    for (let index = 0; index < signed.length; index += 1) {
      const code = signed[index];
      values[index] = code === 0 ? 0 : Math.sign(code) * floor * 10 ** (((Math.abs(code) - 1) / 126) * decades);
    }
    return values;
  }
  return Float32Array.from(state.hidden ?? []);
}

export function decodeColumn<T extends Uint8Array | Uint32Array | Float32Array>(
  encoded: string,
  kind: new (buffer: ArrayBuffer) => T,
): T {
  const bytes = Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0));
  return new kind(bytes.buffer);
}

export function isWholeBrain(graph: GraphPayload): graph is WholeBrainPayload {
  return graph.format === "columnar-v1";
}
