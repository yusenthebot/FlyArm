import { decodeColumn, type WholeBrainPayload } from "../types";

export interface WholeBrainGeometry {
  ids: Uint32Array;
  stateIndex: Uint32Array;
  positions: Float32Array;
  roles: Uint8Array;
  contextEdges: Float32Array;
  slotById: Map<number, number>;
}

export function decodeGeometry(graph: WholeBrainPayload): WholeBrainGeometry {
  const ids = decodeColumn(graph.ids, Uint32Array);
  const positions = decodeColumn(graph.positions, Float32Array);
  const pairs = decodeColumn(graph.context_edges, Uint32Array);
  const contextEdges = new Float32Array(pairs.length * 3);
  for (let index = 0; index < pairs.length; index += 1) {
    contextEdges.set(positions.subarray(pairs[index] * 3, pairs[index] * 3 + 3), index * 3);
  }
  const slotById = new Map<number, number>();
  ids.forEach((id, slot) => slotById.set(id, slot));
  return {
    ids,
    stateIndex: decodeColumn(graph.state_index, Uint32Array),
    positions,
    roles: decodeColumn(graph.roles, Uint8Array),
    contextEdges,
    slotById,
  };
}
