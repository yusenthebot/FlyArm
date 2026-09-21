import type { CausalMode } from "../types";

const MODES: { value: CausalMode; label: string }[] = [
  { value: "connectome", label: "MaleCNS" },
  { value: "shuffled", label: "Shuffled" },
  { value: "edges_off", label: "Edges off" },
];

export function Toolbar({
  running,
  connected,
  mode,
  onReset,
  onRun,
  onPause,
  onStep,
  onMode,
}: {
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
        <p>MaleCNS v1.0 · 256 measured neurons · 4,678 synaptic edges</p>
      </div>
      <div className="transport" aria-label="Simulation controls">
        <button onClick={onReset}>↻ <span>Reset</span></button>
        <button className={running ? "active" : ""} onClick={onRun}>▶ <span>Run</span></button>
        <button onClick={onPause}>Ⅱ <span>Pause</span></button>
        <button onClick={onStep}>▷| <span>Step</span></button>
      </div>
      <fieldset className="mode-switch">
        <legend>Causal mode</legend>
        {MODES.map((item) => (
          <button
            key={item.value}
            aria-pressed={mode === item.value}
            onClick={() => onMode(item.value)}
          >
            {item.label}
          </button>
        ))}
      </fieldset>
      <span className={`connection ${connected ? "online" : ""}`}>
        {connected ? "LIVE" : "OFFLINE"}
      </span>
    </header>
  );
}
