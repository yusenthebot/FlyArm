const JOINTS = ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "f1", "f2"];
const WIDTH = 216;
const LABEL = 22;
const ROW = 15;

/** The newest action chunk: one sparkline per joint velocity in [-1, 1], now at the left. */
export function MotorPlan({ plan, dt }: { plan: number[][]; dt: number }) {
  const steps = plan.length;
  const span = WIDTH - LABEL - 6;
  const x = (step: number) => LABEL + (span * step) / Math.max(steps - 1, 1);
  const height = JOINTS.length * ROW;
  return (
    <figure className="motor-plan" aria-label="Predicted action chunk">
      <figcaption>
        <span>Motor plan</span>
        <span>next {(steps * dt).toFixed(1)} s · {steps} steps</span>
      </figcaption>
      <svg viewBox={`0 0 ${WIDTH} ${height}`} width={WIDTH} height={height} role="img">
        {JOINTS.map((joint, index) => {
          const middle = index * ROW + ROW / 2;
          const y = (value: number) => middle - (Math.max(-1, Math.min(1, value)) * (ROW - 3)) / 2;
          const points = plan.map((row, step) => `${x(step).toFixed(1)},${y(row[index]).toFixed(1)}`);
          return (
            <g key={joint}>
              <text x={0} y={middle + 3}>{joint}</text>
              <line x1={LABEL} x2={WIDTH - 6} y1={middle} y2={middle} className="plan-zero" />
              <polyline points={points.join(" ")} className="plan-line" />
              <circle cx={x(0)} cy={y(plan[0][index])} r={2} className="plan-now" />
            </g>
          );
        })}
      </svg>
    </figure>
  );
}
