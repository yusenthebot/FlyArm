import { useEffect, useMemo, useRef } from "react";
import * as THREE from "three";

import type { RobotPayload } from "../types";

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

function buildGeoms(robot: RobotPayload): BuiltGeom[] {
  return robot.geoms.map((geom) => {
    let geometry: THREE.BufferGeometry;
    if (geom.kind === "mesh") {
      geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(bytes(geom.positions)), 3));
      geometry.setAttribute("normal", new THREE.BufferAttribute(new Int8Array(bytes(geom.normals)), 3, true));
      const index = geom.index_width === 2 ? new Uint16Array(bytes(geom.index)) : new Uint32Array(bytes(geom.index));
      geometry.setIndex(new THREE.BufferAttribute(index, 1));
    } else {
      geometry = new THREE.BoxGeometry(...geom.size);
    }
    const [r, g, b, a] = geom.rgba;
    return {
      body: geom.body,
      geometry,
      color: new THREE.Color().setRGB(r, g, b, THREE.SRGBColorSpace),
      opacity: a,
      position: new THREE.Vector3(...geom.pos),
      quaternion: toQuaternion(geom.quat),
    };
  });
}

/**
 * The Menagerie Panda exactly as MuJoCo compiled it: one group per body, posed every frame
 * from the simulator's body positions and quaternions. Children use MuJoCo's z-up world
 * frame; the parent group turns it into the scene's y-up frame, matching toScene().
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
          key={name}
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
              >
                <meshStandardMaterial
                  color={geom.color}
                  roughness={0.42}
                  metalness={0.08}
                  transparent={geom.opacity < 1}
                  opacity={geom.opacity}
                />
              </mesh>
            ))}
        </group>
      ))}
    </group>
  );
}
