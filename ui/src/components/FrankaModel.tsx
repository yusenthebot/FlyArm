import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { RobotGeom, RobotPayload } from "../types";

function bytes(encoded: string): ArrayBuffer {
  return Uint8Array.from(atob(encoded), (char) => char.charCodeAt(0)).buffer;
}

interface BuiltGeom {
  body: number;
  geometry: THREE.BufferGeometry;
  color: THREE.Color;
  opacity: number;
  position: THREE.Vector3;
  quaternion: THREE.Quaternion;
}

/** MuJoCo quaternions are [w, x, y, z]; three.js expects (x, y, z, w). */
const toQuaternion = ([w, x, y, z]: number[]) => new THREE.Quaternion(x, y, z, w);

/** three.js cylinders and capsules run along +Y; MuJoCo's run along +Z. */
function alongZ(geometry: THREE.BufferGeometry): THREE.BufferGeometry {
  return geometry.rotateX(Math.PI / 2);
}

function buildGeometry(geom: RobotGeom): THREE.BufferGeometry {
  if (geom.kind === "mesh") {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bytes(geom.positions)), 3));
    geometry.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(bytes(geom.normals)), 3, true));
    const index = geom.index_width === 2 ? new Uint16Array(bytes(geom.index)) : new Uint32Array(bytes(geom.index));
    geometry.setIndex(new THREE.BufferAttribute(index, 1));
    return geometry;
  }
  const [a, b, c] = geom.size;
  switch (geom.kind) {
    case "box":
      return new THREE.BoxGeometry(2 * a, 2 * b, 2 * c);
    case "sphere":
      return new THREE.SphereGeometry(a, 24, 16);
    case "cylinder":
      return alongZ(new THREE.CylinderGeometry(a, a, 2 * b, 28));
    case "capsule":
      return alongZ(new THREE.CapsuleGeometry(a, 2 * b, 8, 20));
    case "plane":
      // A zero plane size means infinite in MuJoCo; draw a generous finite floor.
      return new THREE.PlaneGeometry(2 * (a || 5), 2 * (b || 5));
  }
}

function buildGeoms(robot: RobotPayload): BuiltGeom[] {
  return robot.geoms.map((geom) => {
    const [r, g, b, a] = geom.rgba;
    return {
      body: geom.body,
      geometry: buildGeometry(geom),
      color: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace),
      opacity: a,
      position: new THREE.Vector3(...geom.pos),
      quaternion: toQuaternion(geom.quat),
    };
  });
}

/**
 * Bodies exactly as MuJoCo compiled them, each group posed every frame from the simulator's
 * body positions and quaternions. Children use MuJoCo's z-up world frame; the parent group
 * turns it into the scene's y-up frame, matching toScene() in RobotView.
 */
export function FrankaModel({ robot, poses }: { robot: RobotPayload; poses: number[][] }) {
  const geoms = useMemo(() => buildGeoms(robot), [robot]);
  const bodies = useRef<(THREE.Group | null)[]>([]);

  useEffect(() => () => geoms.forEach((geom) => geom.geometry.dispose()), [geoms]);

  useEffect(() => {
    poses.forEach((pose, index) => {
      const group = bodies.current[index];
      if (!group || pose.length !== 7) return;
      group.position.set(pose[0], pose[1], pose[2]);
      group.quaternion.copy(toQuaternion(pose.slice(3)));
    });
  }, [poses]);

  return (
    <group rotation={[-Math.PI / 2, 0, 0]}>
      {robot.bodies.map((name, bodyIndex) => (
        <group
          key={`${bodyIndex}-${name}`}
          ref={(group) => {
            bodies.current[bodyIndex] = group;
          }}
        >
          {geoms
            .filter((geom) => geom.body === bodyIndex)
            .map((geom, geomIndex) => (
              <mesh
                key={geomIndex}
                geometry={geom.geometry}
                position={geom.position}
                quaternion={geom.quaternion}
                receiveShadow={false}
              >
                <meshStandardMaterial
                  color={geom.color}
                  roughness={0.5}
                  metalness={0.06}
                  transparent={geom.opacity < 1}
                  opacity={geom.opacity}
                  side={geom.opacity < 1 ? THREE.DoubleSide : THREE.FrontSide}
                />
              </mesh>
            ))}
        </group>
      ))}
    </group>
  );
}
