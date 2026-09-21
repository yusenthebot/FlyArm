import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import { useEffect, useState } from "react";

import { loadRobot } from "../api";
import type { RobotPayload, RuntimeState } from "../types";
import { FrankaModel } from "./FrankaModel";
import { MotorPlan } from "./MotorPlan";

// FrankaKitchen's default free camera (lookat, distance 2.2, azimuth 70, elevation -35),
// converted from MuJoCo z-up coordinates (x, y, z) to the scene's y-up frame (x, z, -y).
const LOOKAT: [number, number, number] = [-0.2, 2.0, -0.5];
const CAMERA: [number, number, number] = [-0.82, 3.26, 1.19];

export function KitchenView({ state }: { state: RuntimeState }) {
  const [scene, setScene] = useState<RobotPayload | null>(null);
  useEffect(() => {
    let current = true;
    loadRobot()
      .then((payload) => {
        if (current) setScene(payload);
      })
      .catch(() => {
        if (current) setScene(null);
      });
    return () => {
      current = false;
    };
  }, []);
  const tasks = state.tasks ?? [];
  const completed = new Set(state.completed ?? []);
  return (
    <div className="viewport robot-viewport">
      <Canvas camera={{ position: CAMERA, fov: 45 }} dpr={[1, 1.5]}>
        <color attach="background" args={["#f4f7f8"]} />
        <ambientLight intensity={1.1} />
        <directionalLight position={[1.5, 4.0, 2.0]} intensity={2.0} />
        <directionalLight position={[-2.0, 3.0, -1.0]} intensity={0.8} />
        {scene && state.robot_bodies && <FrankaModel robot={scene} poses={state.robot_bodies} />}
        <OrbitControls makeDefault target={LOOKAT} minDistance={0.8} maxDistance={6} />
      </Canvas>
      <dl className="sim-readout">
        <div><dt>Step</dt><dd>{state.step} / 280</dd></div>
        <div><dt>D4RL score</dt><dd>{(state.score ?? 0).toFixed(0)}</dd></div>
        {tasks.map((task) => (
          <div key={task} className={completed.has(task) ? "task-done" : ""}>
            <dt>{task}</dt>
            <dd>
              {completed.has(task) ? "done" : `distance ${(state.task_distance?.[task] ?? 0).toFixed(2)}`}
            </dd>
          </div>
        ))}
      </dl>
      {state.plan && state.plan.length > 1 && (
        <MotorPlan plan={state.plan} dt={state.control_dt ?? 0.08} />
      )}
      <div className="viewport-caption">
        <span>
          D4RL FrankaKitchen · task done when distance &lt; 0.30
          {scene && " · scene meshes: Gymnasium-Robotics FrankaKitchen-v1"}
        </span>
        <span>Drag · orbit / Scroll · zoom</span>
      </div>
    </div>
  );
}
