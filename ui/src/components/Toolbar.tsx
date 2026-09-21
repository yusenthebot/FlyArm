import type { CausalMode } from "../types";

const LABELS: Record<CausalMode, string> = {
  connectome: "MaleCNS",
  shuffled: "Shuffled",
  edges_off: "Edges off",
  direct_only: "Direct I/O only",
};

export function Toolbar({
  title,
  modes,
  running,
  connected,
  mode,
  onReset,
  onRun,
  onPause,
  onStep,
  onMode,
}: {
  title: string;
  modes: CausalMode[];
  running: boolean;
  connected: boolean;
  mode: CausalMode;
  onReset: () => void;
  onRun: () => void;
  onPause: () => void;
  onStep: () => void;
  onMode: (mode: CausalMode) => void;
}) {
  return (
    <header className="toolbar">
      <div className="brand">
        <h1>FlyArm</h1>
        <p>MaleCNS v1.0 · {title}</p>
      </div>
      <div className="transport" aria-label="Simulation controls">
        <button onClick={onReset}>↻ <span>Reset</span></button>
        <button className={running ? "active" : ""} onClick={onRun}>▶ <span>Run</span></button>
        <button onClick={onPause}>Ⅱ <span>Pause</span></button>
        <button onClick={onStep}>▷| <span>Step</span></button>
      </div>
      <fieldset className="mode-switch">
        <legend>Causal mode</legend>
        {modes.map((value) => (
          <button key={value} aria-pressed={mode === value} onClick={() => onMode(value)}>
            {LABELS[value]}
          </button>
        ))}
      </fieldset>
      <span className={`connection ${connected ? "online" : ""}`}>
        {connected ? "LIVE" : "OFFLINE"}
      </span>
    </header>
  );
}
