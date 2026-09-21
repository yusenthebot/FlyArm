import { OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { NeuronDetail, WholeBrainPayload } from "../types";
import type { WholeBrainGeometry } from "./wholeBrainGeometry";

const ROLE_RGB: [number, number, number][] = [
  [0.965, 0.416, 0.31], // internal, coral
  [0.957, 0.867, 0.263], // input (or first input group), yellow
  [0.259, 0.875, 0.957], // output, cyan
  [0.584, 0.463, 0.953], // second input group, violet
];
const ROLE_HEX = ["#f66a4f", "#f4dd43", "#42dff4", "#9576f3"];
const DEFAULT_LABELS: Record<string, string> = {
  "1": "input · ascending",
  "0": "internal",
  "2": "output · descending + motor",
};
const IDLE: [number, number, number] = [0.78, 0.83, 0.86];
// Rate states span four decades (inputs near 1, most internal neurons near 1e-3), so
// brightness follows log10 |state| on a fixed 1e-4..1 scale, identical in every frame.
const FLOOR = 1e-4;
const DECADES = -Math.log10(FLOOR);

function NeuronCloud({
  geometry,
  hidden,
  onSelect,
}: {
  geometry: WholeBrainGeometry;
  hidden: Float32Array;
  onSelect: (id: number) => void;
}) {
  const colorRef = useRef<THREE.BufferAttribute>(null);
  const initialColors = useMemo(() => new Float32Array(geometry.ids.length * 3), [geometry]);

  useEffect(() => {
    const attribute = colorRef.current;
    if (!attribute) return;
    const colors = attribute.array as Float32Array;
    const { roles, stateIndex } = geometry;
    for (let slot = 0; slot < stateIndex.length; slot += 1) {
      const value = hidden[stateIndex[slot]] ?? 0;
      const size = Math.abs(value);
      const magnitude = size < FLOOR ? 0 : Math.min(Math.log10(size / FLOOR) / DECADES, 1);
      const role = ROLE_RGB[roles[slot]];
      for (let channel = 0; channel < 3; channel += 1) {
        colors[slot * 3 + channel] = IDLE[channel] + (role[channel] - IDLE[channel]) * magnitude;
      }
    }
    attribute.needsUpdate = true;
  }, [geometry, hidden]);

  return (
    <points
      onClick={(event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        if (event.index !== undefined) onSelect(geometry.ids[event.index]);
      }}
    >
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[geometry.positions, 3]} />
        <bufferAttribute ref={colorRef} attach="attributes-color" args={[initialColors, 3]} />
      </bufferGeometry>
      <pointsMaterial size={0.011} vertexColors sizeAttenuation transparent opacity={0.9} depthWrite={false} />
    </points>
  );
}

function Segments({ points, color, opacity }: { points: Float32Array; color: string; opacity: number }) {
  if (points.length === 0) return null;
  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[points, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color={color} transparent opacity={opacity} depthWrite={false} />
    </lineSegments>
  );
}

function Selection({ geometry, detail }: { geometry: WholeBrainGeometry; detail: NeuronDetail | null }) {
  const lines = useMemo(() => {
    if (!detail) return { up: new Float32Array(), down: new Float32Array(), at: null };
    const slot = geometry.slotById.get(detail.id);
    const at = slot === undefined ? null : geometry.positions.subarray(slot * 3, slot * 3 + 3);
    const build = (partners: NeuronDetail["upstream"]) => {
      if (!at) return new Float32Array();
      const values: number[] = [];
      for (const partner of partners) {
        const other = geometry.slotById.get(partner.id);
        if (other === undefined) continue;
        values.push(...at, ...geometry.positions.subarray(other * 3, other * 3 + 3));
      }
      return new Float32Array(values);
    };
    return { up: build(detail.upstream), down: build(detail.downstream), at };
  }, [detail, geometry]);
  return (
    <>
      <Segments points={lines.up} color="#9a5a00" opacity={0.85} />
      <Segments points={lines.down} color="#0d5f7a" opacity={0.85} />
      {lines.at && (
        <mesh position={[lines.at[0], lines.at[1], lines.at[2]]}>
          <sphereGeometry args={[0.02, 12, 12]} />
          <meshBasicMaterial color="#10212c" transparent opacity={0.85} />
        </mesh>
      )}
    </>
  );
}

export function WholeBrainView({
  graph,
  geometry,
  hidden,
  detail,
  onSelect,
}: {
  graph: WholeBrainPayload;
  geometry: WholeBrainGeometry;
  hidden: Float32Array;
  detail: NeuronDetail | null;
  onSelect: (id: number) => void;
}) {
  const undrawn = graph.neurons - graph.drawn;
  return (
    <div className="viewport connectome-viewport">
      <div className="legend" aria-label="Neuron role legend">
        {Object.entries(graph.role_labels ?? DEFAULT_LABELS)
          .sort(([a], [b]) => (a === "0" ? 1 : b === "0" ? -1 : Number(a) - Number(b)))
          .map(([role, label]) => (
            <span key={role}><i style={{ background: ROLE_HEX[Number(role)] ?? "#999" }} /> {label}</span>
          ))}
        <span><i style={{ background: "#c7d4db" }} /> brightness: log |state|, 1e-4 to 1</span>
      </div>
      <Canvas camera={{ position: [0, 0.1, 2.4], fov: 42 }} dpr={[1, 1.5]} raycaster={{ params: { Points: { threshold: 0.008 } } as THREE.Raycaster["params"] }}>
        <color attach="background" args={["#f8fbfc"]} />
        <Segments points={geometry.contextEdges} color="#7898aa" opacity={0.035} />
        <NeuronCloud geometry={geometry} hidden={hidden} onSelect={onSelect} />
        <Selection geometry={geometry} detail={detail} />
        <OrbitControls makeDefault enableDamping minDistance={0.4} maxDistance={5} />
      </Canvas>
      <div className="viewport-caption">
        <span>
          {graph.drawn.toLocaleString()} of {graph.neurons.toLocaleString()} neurons drawn ·{" "}
          {undrawn.toLocaleString()} without a location simulated, not drawn
          {graph.proxy_positioned_afferents
            ? ` · ${graph.proxy_positioned_afferents.toLocaleString()} sensory afferents (soma outside the CNS) placed at their partners' contact-weighted centroid`
            : ""}
        </span>
        <span>Faint lines: {graph.context_edge_note}</span>
      </div>
    </div>
  );
}
