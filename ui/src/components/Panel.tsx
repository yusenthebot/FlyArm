import type { PropsWithChildren, ReactNode } from "react";

interface PanelProps extends PropsWithChildren {
  title: string;
  actions?: ReactNode;
  className?: string;
}

export function Panel({ title, actions, className = "", children }: PanelProps) {
  return (
    <section className={`panel ${className}`}>
      <header className="panel-header">
        <h2>{title}</h2>
        {actions}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  );
}
