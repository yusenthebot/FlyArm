import { useMemo } from "react";

import type { TimelineSample } from "../types";

const WIDTH = 1200;
const HEIGHT = 185;

function pathFor(samples: TimelineSample[], accessor: (sample: TimelineSample) => number, min: number, max: number, y0: number, height: number) {
  if (samples.length < 2) return "";
  const range = Math.max(max - min, 1e-6);
  return samples
    .map((sample, index) => {
      const x = (index / (samples.length - 1)) * WIDTH;
      const y = y0 + height - ((accessor(sample) - min) / range) * height;
      return `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

export function Timeline({ samples, kitchen = false }: { samples: TimelineSample[]; kitchen?: boolean }) {
  const visible = samples.slice(-240);
  const paths = useMemo(
    () => ({
      dx: pathFor(visible, (s) => s.action[0], -1, 1, 8, 42),
      dy: pathFor(visible, (s) => s.action[1], -1, 1, 8, 42),
      dz: pathFor(visible, (s) => s.action[2], -1, 1, 8, 42),
      grip: pathFor(visible, (s) => s.action[3], -1, 1, 8, 42),
      height: pathFor(visible, (s) => s.objectHeight, 0, kitchen ? 1 : 0.24, 62, 30),
      error: pathFor(visible, (s) => s.goalError, 0, 0.3, 104, 28),
      activation: pathFor(visible, (s) => s.activation, 0, 1, 144, 28),
    }),
    [visible, kitchen],
  );
  return (
    <div className="timeline-wrap">
      <div className="timeline-labels" aria-hidden>
        <span>{kitchen ? "Joints 1-4" : "Action"}</span>
        <span>{kitchen ? "Tasks done" : "Object height"}</span>
        <span>{kitchen ? "" : "Goal error"}</span>
        <span>Neural |mean|</span>
      </div>
      <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} preserveAspectRatio="none" role="img" aria-label="Live synchronized simulation timeline">
        <defs>
          <pattern id="grid" width="100" height="36" patternUnits="userSpaceOnUse">
            <path d="M100 0H0V36" fill="none" stroke="#1a3549" strokeWidth="1" />
          </pattern>
        </defs>
        <rect width={WIDTH} height={HEIGHT} fill="url(#grid)" />
        <path d={paths.dx} className="signal dx" />
        <path d={paths.dy} className="signal dy" />
        <path d={paths.dz} className="signal dz" />
        <path d={paths.grip} className="signal grip" />
        <path d={paths.height} className="signal height" />
        <path d={paths.error} className="signal error" />
        <path d={paths.activation} className="signal activation" />
        {visible.map((sample, index) => {
          if (!sample.left && !sample.right) return null;
          const x = (index / Math.max(visible.length - 1, 1)) * WIDTH;
          return <line key={sample.step} x1={x} x2={x} y1={176} y2={183} className={sample.left && sample.right ? "contact both" : "contact"} />;
        })}
      </svg>
      <div className="timeline-legend">
        {kitchen ? (
          <><span className="c-dx">j1</span><span className="c-dy">j2</span><span className="c-dz">j3</span><span className="c-grip">j4</span><span>joint velocity commands</span></>
        ) : (
          <><span className="c-dx">dx</span><span className="c-dy">dy</span><span className="c-dz">dz</span><span className="c-grip">grip</span><span>contacts at baseline</span></>
        )}
      </div>
    </div>
  );
}
