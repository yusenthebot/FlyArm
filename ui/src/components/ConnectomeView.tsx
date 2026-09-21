import { Line, OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useMemo } from "react";

import type { GraphNode, SubgraphPayload } from "../types";

const ROLE_COLOR = {
  input: "#f4dd43",
  internal: "#f66a4f",
  output: "#42dff4",
} as const;

function NodePoint({
  node,
  activation,
  selected,
  onSelect,
}: {
  node: GraphNode;
  activation: number;
  selected: boolean;
  onSelect: (id: number) => void;
}) {
  const magnitude = Math.min(Math.abs(activation), 1);
  return (
    <mesh
      position={node.position}
      scale={selected ? 1.8 : 0.8 + magnitude * 0.8}
      onClick={(event) => {
        event.stopPropagation();
        onSelect(node.id);
      }}
    >
      <sphereGeometry args={[selected ? 0.034 : 0.022, 10, 10]} />
      <meshBasicMaterial
        color={ROLE_COLOR[node.role]}
        transparent
        opacity={0.62 + magnitude * 0.38}
      />
    </mesh>
  );
}

function GraphScene({
  graph,
  hidden,
  selectedId,
  onSelect,
}: {
  graph: SubgraphPayload;
  hidden: Float32Array;
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  const byId = useMemo(() => new Map(graph.nodes.map((node) => [node.id, node])), [graph]);
  const baseSegments = useMemo(
    () =>
      graph.edges.flatMap((edge) => {
        const source = byId.get(edge.source);
        const target = byId.get(edge.target);
        return source && target ? [source.position, target.position] : [];
      }),
    [byId, graph.edges],
  );
  const selectedSegments = useMemo(() => {
    if (selectedId === null) return [];
    return graph.edges.flatMap((edge) => {
      if (edge.source !== selectedId && edge.target !== selectedId) return [];
      const source = byId.get(edge.source);
      const target = byId.get(edge.target);
      return source && target ? [source.position, target.position] : [];
    });
  }, [byId, graph.edges, selectedId]);

  return (
    <>
      <ambientLight intensity={0.8} />
      <Line segments points={baseSegments} color="#7898aa" transparent opacity={0.2} />
      {selectedSegments.length > 0 && (
        <Line segments points={selectedSegments} color="#193a50" transparent opacity={0.82} />
      )}
      {graph.nodes.map((node) => (
        <NodePoint
          key={node.id}
          node={node}
          activation={hidden[node.index] ?? 0}
          selected={node.id === selectedId}
          onSelect={onSelect}
        />
      ))}
      <OrbitControls makeDefault enableDamping minDistance={1.2} maxDistance={5} />
    </>
  );
}

export function ConnectomeView({
  graph,
  hidden,
  selectedId,
  onSelect,
}: {
  graph: SubgraphPayload;
  hidden: Float32Array;
  selectedId: number | null;
  onSelect: (id: number) => void;
}) {
  return (
    <div className="viewport connectome-viewport">
      <div className="legend" aria-label="Neuron role legend">
        {Object.entries(ROLE_COLOR).map(([label, color]) => (
          <span key={label}>
            <i style={{ background: color }} /> {label}
          </span>
        ))}
      </div>
      <Canvas camera={{ position: [0, 0.25, 2.55], fov: 42 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#f8fbfc"]} />
        <GraphScene
          graph={graph}
          hidden={hidden}
          selectedId={selectedId}
          onSelect={onSelect}
        />
      </Canvas>
      <div className="viewport-caption">
        <span>Measured soma coordinates</span>
        <span>Drag · orbit / Scroll · zoom</span>
      </div>
    </div>
  );
}
