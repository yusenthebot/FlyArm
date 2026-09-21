export type CausalMode = "connectome" | "shuffled" | "edges_off";

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

export interface GraphPayload {
  nodes: GraphNode[];
  edges: GraphEdge[];
  graph_fingerprint: string;
  interface_fingerprint: string;
  dataset: string;
  layout: "measured_soma_locations";
}

export interface EvidenceSummary {
  graph_mediated: "supported" | "not_supported" | "pending";
  topology_advantage: "supported" | "not_supported" | "pending";
  connectome_success_rate: number | null;
  shuffled_success_rate: number | null;
  edges_off_success_rate: number | null;
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
  gripper_points: [number, number, number][];
  action: [number, number, number, number];
  hidden: number[];
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
  hidden: [],
  evidence: {
    graph_mediated: "pending",
    topology_advantage: "pending",
    connectome_success_rate: null,
    shuffled_success_rate: null,
    edges_off_success_rate: null,
  },
};
