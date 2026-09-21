import { useCallback, useEffect, useMemo, useState } from "react";

import { controls, loadGraph, loadNeuron, openStateStream } from "./api";
import { AuditPanel, type SelectedNeuron } from "./components/AuditPanel";
import { ConnectomeView } from "./components/ConnectomeView";
import { Panel } from "./components/Panel";
import { RobotView } from "./components/RobotView";
import { Timeline } from "./components/Timeline";
import { Toolbar } from "./components/Toolbar";
import { WholeBrainView } from "./components/WholeBrainView";
import { decodeGeometry } from "./components/wholeBrainGeometry";
import {
  decodeActivity,
  EMPTY_STATE,
  isWholeBrain,
  type CausalMode,
  type GraphPayload,
  type NeuronDetail,
  type RuntimeState,
  type TimelineSample,
} from "./types";

const DEFAULT_MODES: CausalMode[] = ["connectome", "shuffled", "edges_off"];

function meanAbsolute(values: Float32Array): number {
  let total = 0;
  for (const value of values) total += Math.abs(value);
  return values.length ? total / values.length : 0;
}

function App() {
  const [graph, setGraph] = useState<GraphPayload | null>(null);
  const [state, setState] = useState<RuntimeState>(EMPTY_STATE);
  const [hidden, setHidden] = useState<Float32Array>(() => new Float32Array());
  const [timeline, setTimeline] = useState<TimelineSample[]>([]);
  const [chosenId, setSelectedId] = useState<number | null>(null);
  const [detail, setDetail] = useState<NeuronDetail | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadGraph()
      .then((payload) => {
        setGraph(payload);
        if (!isWholeBrain(payload)) {
          setSelectedId(payload.nodes.find((node) => node.role === "input")?.id ?? payload.nodes[0]?.id ?? null);
        }
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
  }, []);

  const geometry = useMemo(() => (graph && isWholeBrain(graph) ? decodeGeometry(graph) : null), [graph]);

  // Until the user picks one, show the first drawn ascending (input) neuron.
  const selectedId = useMemo(() => {
    if (chosenId !== null || !geometry) return chosenId;
    const slot = geometry.roles.indexOf(1);
    return slot >= 0 ? geometry.ids[slot] : null;
  }, [chosenId, geometry]);

  useEffect(() => {
    if (!graph || !isWholeBrain(graph) || selectedId === null) return;
    let current = true;
    loadNeuron(selectedId)
      .then((value) => {
        if (current) setDetail(value);
      })
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)));
    return () => {
      current = false;
    };
  }, [graph, selectedId]);

  useEffect(
    () =>
      openStateStream(
        (next) => {
          const activity = decodeActivity(next);
          setState({ ...next, connected: true });
          setHidden(activity);
          setTimeline((current) => [
            ...current.slice(-479),
            {
              step: next.step,
              time: next.sim_time,
              action: next.action,
              objectHeight: next.object_height,
              goalError: next.goal_error,
              activation: meanAbsolute(activity),
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
  const whole = graph !== null && isWholeBrain(graph);
  const selected = useMemo<SelectedNeuron | null>(() => {
    if (!graph) return null;
    if (isWholeBrain(graph)) return detail;
    return graph.nodes.find((node) => node.id === selectedId) ?? null;
  }, [detail, graph, selectedId]);

  if (!graph) {
    return <main className="loading"><h1>FlyArm</h1><p>{error ?? "Loading measured MaleCNS graph…"}</p></main>;
  }
  return (
    <main className="app-shell">
      <Toolbar
        title={graph.title ?? "256 measured neurons · 4,678 synaptic edges"}
        modes={graph.modes ?? DEFAULT_MODES}
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
        <Panel title={whole ? "LIVE COMPLETE CONNECTOME" : "LIVE CONNECTOME"} className="connectome-panel">
          {isWholeBrain(graph) ? (
            geometry && (
              <WholeBrainView graph={graph} geometry={geometry} hidden={hidden} detail={detail} onSelect={setSelectedId} />
            )
          ) : (
            <ConnectomeView graph={graph} hidden={hidden} selectedId={selectedId} onSelect={setSelectedId} />
          )}
        </Panel>
        <Panel title="MUJOCO PICK + PLACE" className="robot-panel">
          <RobotView state={state} onMove={move} />
        </Panel>
        <AuditPanel graph={graph} selected={selected} detail={whole ? detail : null} hidden={hidden} state={state} />
      </div>
      <Panel title="SYNCHRONIZED LIVE TIMELINE" className="timeline-panel">
        <Timeline samples={timeline} />
      </Panel>
      <footer>
        <span>MaleCNS v1.0 · CC BY 4.0 · source hashes verified</span>
        <span>
          {whole
            ? "Complete measured connectome · abstract rate dynamics · not a physiological emulation"
            : "Measured connectome · abstract dynamics · no whole-brain emulation claim"}
        </span>
      </footer>
    </main>
  );
}

export default App;
