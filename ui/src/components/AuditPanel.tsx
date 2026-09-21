import type { GraphPayload, NeuronDetail, Partner, RuntimeState } from "../types";

/** The fields both the subgraph node list and the whole-brain detail endpoint provide. */
export interface SelectedNeuron {
  id: number;
  index: number;
  type: string | null;
  superclass: string | null;
  role: "input" | "internal" | "output";
  sign: -1 | 0 | 1;
  in_degree: number;
  out_degree: number;
}

function metric(value: number | null | undefined) {
  return value === null || value === undefined ? "pending" : `${(value * 100).toFixed(0)}%`;
}

function Partners({ label, partners }: { label: string; partners: Partner[] }) {
  return (
    <div className="partners">
      <h3>{label}</h3>
      <ol>
        {partners.slice(0, 6).map((partner) => (
          <li key={partner.id}>
            <span>{partner.type ?? "untyped"}</span>
            <b>{partner.contacts}</b>
          </li>
        ))}
      </ol>
    </div>
  );
}

export function AuditPanel({
  graph,
  selected,
  detail,
  hidden,
  state,
}: {
  graph: GraphPayload;
  selected: SelectedNeuron | null;
  detail: NeuronDetail | null;
  hidden: Float32Array;
  state: RuntimeState;
}) {
  const activation = selected ? (hidden[selected.index] ?? 0) : 0;
  const evidence = state.evidence;
  return (
    <div className="audit-stack">
      <section className="audit-section">
        <h2>NEURON AUDIT</h2>
        {selected ? (
          <dl className="audit-grid">
            <dt>Body ID</dt><dd>{selected.id}</dd>
            <dt>Annotated type</dt><dd>{selected.type ?? "untyped"}</dd>
            <dt>Superclass</dt><dd>{selected.superclass ?? "unknown"}</dd>
            <dt>Interface role</dt><dd className={`role-${selected.role}`}>{selected.role}</dd>
            <dt>Connections</dt><dd>{selected.in_degree} in / {selected.out_degree} out</dd>
            <dt>NT sign</dt><dd>{selected.sign > 0 ? "excitatory proxy" : selected.sign < 0 ? "inhibitory proxy" : "unknown / zero"}</dd>
            <dt>Live activation</dt><dd className="activation-value">{activation.toFixed(4)}</dd>
          </dl>
        ) : (
          <p className="empty-audit">Select a neuron in the measured soma map.</p>
        )}
        {detail && (
          <div className="partner-grid">
            <Partners label="Strongest inputs · contacts" partners={detail.upstream} />
            <Partners label="Strongest outputs · contacts" partners={detail.downstream} />
          </div>
        )}
        <div className="hash-row">
          <span>Graph SHA</span>
          <code title={graph.graph_fingerprint}>{graph.graph_fingerprint.slice(0, 12)}…</code>
          <b>verified</b>
        </div>
      </section>
      <section className="audit-section causal-section">
        <h2>CAUSAL TEST</h2>
        <dl className="audit-grid compact">
          <dt>Current mode</dt><dd>{state.mode}</dd>
          <dt>Contact L / R</dt><dd>{Number(state.contact_left)} / {Number(state.contact_right)}</dd>
          <dt>Grasp history</dt><dd>{state.ever_grasped ? "yes" : "no"}</dd>
          <dt>Lift history</dt><dd>{state.ever_lifted ? "yes" : "no"}</dd>
          <dt>Released success</dt><dd>{state.success ? "yes" : "no"}</dd>
        </dl>
        <div className={`evidence evidence-${evidence.graph_mediated}`}>
          <strong>Graph-mediated: {evidence.graph_mediated.replace("_", " ")}</strong>
          <span>
            MaleCNS {metric(evidence.connectome_success_rate)} · edges off {metric(evidence.edges_off_success_rate)}
            {evidence.direct_only_success_rate !== undefined &&
              ` · direct I/O only ${metric(evidence.direct_only_success_rate)}`}
          </span>
        </div>
        <div className={`evidence evidence-${evidence.topology_advantage}`}>
          <strong>Topology advantage: {evidence.topology_advantage.replace("_", " ")}</strong>
          <span>
            degree-preserving shuffle {metric(evidence.shuffled_success_rate)}
            {evidence.seeds !== undefined && ` · ${evidence.seeds} seed${evidence.seeds === 1 ? "" : "s"}`}
          </span>
        </div>
      </section>
    </div>
  );
}
