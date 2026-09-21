import { OrbitControls } from "@react-three/drei";
import { Canvas, type ThreeEvent } from "@react-three/fiber";
import { useMemo, useState } from "react";
import * as THREE from "three";

import type { RuntimeState } from "../types";

type Movable = "object" | "goal";

const toScene = ([x, y, z]: [number, number, number]): [number, number, number] => [x, z, -y];

function Segment({ start, end, radius = 0.028 }: { start: number[]; end: number[]; radius?: number }) {
  const transform = useMemo(() => {
    const a = new THREE.Vector3(...toScene(start as [number, number, number]));
    const b = new THREE.Vector3(...toScene(end as [number, number, number]));
    const direction = b.clone().sub(a);
    const quaternion = new THREE.Quaternion().setFromUnitVectors(
      new THREE.Vector3(0, 1, 0),
      direction.clone().normalize(),
    );
    return { midpoint: a.add(b).multiplyScalar(0.5), quaternion, length: direction.length() };
  }, [start, end]);
  return (
    <mesh position={transform.midpoint} quaternion={transform.quaternion}>
      <cylinderGeometry args={[radius, radius, transform.length, 16]} />
      <meshStandardMaterial color="#e7e9e8" metalness={0.18} roughness={0.46} />
    </mesh>
  );
}

function RobotSkeleton({
  points,
  gripperPoints,
}: {
  points: [number, number, number][];
  gripperPoints: [number, number, number][];
}) {
  const hand = points.at(-1);
  return (
    <group>
      {points.slice(0, -1).map((point, index) => (
        <Segment key={index} start={point} end={points[index + 1]} radius={index < 2 ? 0.038 : 0.029} />
      ))}
      {points.map((point, index) => (
        <mesh key={`joint-${index}`} position={toScene(point)}>
          <sphereGeometry args={[index === 0 ? 0.052 : 0.04, 18, 18]} />
          <meshStandardMaterial color={index % 2 ? "#202c36" : "#f0f1ef"} roughness={0.35} />
        </mesh>
      ))}
      {hand && gripperPoints.map((point, index) => (
        <group key={`finger-${index}`}>
          <Segment start={hand} end={point} radius={0.012} />
          <mesh position={toScene(point)}>
            <boxGeometry args={[0.016, 0.055, 0.014]} />
            <meshStandardMaterial color="#161f27" roughness={0.55} />
          </mesh>
        </group>
      ))}
    </group>
  );
}

function RobotScene({
  state,
  onMove,
}: {
  state: RuntimeState;
  onMove: (kind: Movable, value: [number, number, number]) => void;
}) {
  const [dragging, setDragging] = useState<Movable | null>(null);
  const move = (event: ThreeEvent<PointerEvent>) => {
    if (!dragging) return;
    event.stopPropagation();
    const x = THREE.MathUtils.clamp(event.point.x, 0.24, 0.66);
    const y = THREE.MathUtils.clamp(-event.point.z, -0.24, 0.24);
    onMove(dragging, [x, y, dragging === "object" ? 0.02 : 0.002]);
  };
  return (
    <>
      <ambientLight intensity={1.3} />
      <directionalLight position={[1.2, 2.0, 1.0]} intensity={2.2} />
      <gridHelper args={[1.4, 28, "#b8cad4", "#dce6eb"]} position={[0.4, -0.003, 0]} />
      <mesh
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0.4, -0.006, 0]}
        onPointerMove={move}
        onPointerUp={() => setDragging(null)}
      >
        <planeGeometry args={[1.15, 0.82]} />
        <meshStandardMaterial color="#edf3f5" roughness={0.82} />
      </mesh>
      {state.robot_points.length > 1 && (
        <RobotSkeleton points={state.robot_points} gripperPoints={state.gripper_points} />
      )}
      <mesh
        position={toScene(state.object)}
        onPointerDown={(event) => {
          event.stopPropagation();
          setDragging("object");
        }}
      >
        <boxGeometry args={[0.04, 0.04, 0.04]} />
        <meshStandardMaterial color="#f45b43" emissive="#5b1711" />
      </mesh>
      <mesh
        position={toScene(state.goal)}
        rotation={[-Math.PI / 2, 0, 0]}
        onPointerDown={(event) => {
          event.stopPropagation();
          setDragging("goal");
        }}
      >
        <ringGeometry args={[0.024, 0.036, 32]} />
        <meshBasicMaterial color="#49e0b4" side={THREE.DoubleSide} />
      </mesh>
      <OrbitControls makeDefault target={[0.42, 0.24, 0]} minDistance={0.6} maxDistance={2.4} />
    </>
  );
}

export function RobotView({
  state,
  onMove,
}: {
  state: RuntimeState;
  onMove: (kind: Movable, value: [number, number, number]) => void;
}) {
  return (
    <div className="viewport robot-viewport">
      <Canvas camera={{ position: [1.35, 1.08, 1.45], fov: 41 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#f8fbfc"]} />
        <RobotScene state={state} onMove={onMove} />
      </Canvas>
      <dl className="sim-readout">
        <div><dt>Sim time</dt><dd>{state.sim_time.toFixed(2)} s</dd></div>
        <div><dt>Physical state</dt><dd>{state.stage}</dd></div>
        <div><dt>Height</dt><dd>{state.object_height.toFixed(3)} m</dd></div>
        <div><dt>Goal error</dt><dd>{state.goal_error.toFixed(3)} m</dd></div>
      </dl>
      <div className="viewport-caption">
        <span>Drag cube / goal to reset episode</span>
        <span>Drag · orbit / Scroll · zoom</span>
      </div>
    </div>
  );
}
