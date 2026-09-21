import { useCallback, useEffect, useMemo, useState } from "react";

import { controls, loadGraph, openStateStream } from "./api";
import { AuditPanel } from "./components/AuditPanel";
import { ConnectomeView } from "./components/ConnectomeView";
import { Panel } from "./components/Panel";
import { RobotView } from "./components/RobotView";
import { Timeline } from "./components/Timeline";
import { Toolbar } from "./components/Toolbar";
import { EMPTY_STATE, type CausalMode, type GraphPayload, type RuntimeState, type TimelineSample } from "./types";

function App() {
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [state, setState] = useState<RuntimeState>(EMPTY_STATE);
  const [timeline, setTimeline] = useState<TimelineSample[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadGraph()
      .then((payload) => {
        setGraph(payload);
        setSelectedId(payload.nodes.find((node) => node.role === "input")?.id ?? payload.nodes[0]?.id ?? null);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  useEffect(
    () =>
      openStateStream(
        (next) => {
          setState({ ...next, connected: true });
          const activation = next.hidden.length
            ? next.hidden.reduce((sum, value) => sum + Math.abs(value), 0) / next.hidden.length
            : 0;
          setTimeline((current) => [
            ...current.slice(-479),
            {
              step: next.step,
              time: next.sim_time,
              action: next.action,
              objectHeight: next.object_height,
              goalError: next.goal_error,
              activation,
              left: next.contact_left,
              right: next.contact_right,
            },
          ]);
        },
        (connected) => setState((current) => ({ ...current, connected })),
      ),
    [],
  );

  const act = useCallback((operation: () => Promise<void>) => {
    operation().catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);
  const move = useCallback(
    (kind: "object" | "goal", value: [number, number, number]) => {
      const object = kind === "object" ? value : state.object;
      const goal = kind === "goal" ? value : state.goal;
      act(() => controls.reset(object, goal));
      setTimeline([]);
    },
    [act, state.goal, state.object],
  );
  const changeMode = (mode: CausalMode) => {
    act(() => controls.mode(mode));
    setTimeline([]);
  };
  const selected = useMemo(
    () => graph?.nodes.find((node) => node.id === selectedId) ?? null,
    [graph, selectedId],
  );

  if (!graph) {
    return <main className="loading"><h1>FlyArm</h1><p>{error ?? "Loading measured MaleCNS graph…"}</p></main>;
  }
  return (
    <main className="app-shell">
      <Toolbar
        running={state.running}
        connected={state.connected}
        mode={state.mode}
        onReset={() => { act(() => controls.reset()); setTimeline([]); }}
        onRun={() => act(controls.run)}
        onPause={() => act(controls.pause)}
        onStep={() => act(controls.step)}
        onMode={changeMode}
      />
      {error && <button className="error-bar" onClick={() => setError(null)}>{error} · dismiss</button>}
      <div className="workspace">
        <Panel title="LIVE CONNECTOME" className="connectome-panel">
          <ConnectomeView graph={graph} hidden={state.hidden} selectedId={selectedId} onSelect={setSelectedId} />
        </Panel>
        <Panel title="MUJOCO PICK + PLACE" className="robot-panel">
          <RobotView state={state} onMove={move} />
        </Panel>
        <AuditPanel graph={graph} selected={selected} state={state} />
      </div>
      <Panel title="SYNCHRONIZED LIVE TIMELINE" className="timeline-panel">
        <Timeline samples={timeline} />
      </Panel>
      <footer>
        <span>MaleCNS v1.0 · CC BY 4.0 · source hashes verified</span>
        <span>Measured connectome · abstract dynamics · no whole-brain emulation claim</span>
      </footer>
    </main>
  );
}

export default App;
