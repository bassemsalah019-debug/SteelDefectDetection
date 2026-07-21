import type { ReactNode } from "react";
import { CLASS_COLORS } from "./api";

export const Spinner = () => <span className="spinner" aria-label="loading" />;

export function Tile({ label, value, sub }: { label: string; value: ReactNode; sub?: string }) {
  return (
    <div className="tile fade-up">
      <div className="lab">{label}</div>
      <div className="val">{value}</div>
      {sub ? <div className="sub">{sub}</div> : null}
    </div>
  );
}

export function ClassBars({ counts }: { counts: Record<string, number> }) {
  const entries = Object.entries(counts).sort((a, b) => b[1] - a[1]);
  const max = Math.max(1, ...entries.map((e) => e[1]));
  if (!entries.length) return <div className="muted">—</div>;
  return (
    <div className="bars">
      {entries.map(([name, n]) => {
        const col = CLASS_COLORS[name] || "#888";
        const pct = Math.max(4, Math.round((n / max) * 100));
        return (
          <div className="bar-row" key={name}>
            <div className="bar-name">
              <span className="dot" style={{ background: col, boxShadow: `0 0 8px ${col}` }} />
              {name}
            </div>
            <div className="bar-wrap">
              <div className="bar-fill" style={{ width: `${pct}%`, backgroundColor: col, boxShadow: `0 0 14px ${col}88` }} />
            </div>
            <div className="bar-val">{n}</div>
          </div>
        );
      })}
    </div>
  );
}

export function fmtDate(s: string): string {
  try { return new Date(s).toLocaleString(); } catch { return s; }
}

/** Tiny markdown -> HTML for the report (## h2, ### h3, **bold**, - bullets). */
export function ReportText({ text }: { text: string }) {
  const html = text
    .split("\n")
    .map((raw) => {
      let l = raw.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
      l = l.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      if (l.startsWith("### ")) return `<h3>${l.slice(4)}</h3>`;
      if (l.startsWith("## ")) return `<h2>${l.slice(3)}</h2>`;
      if (l.startsWith("- ")) return `<li>${l.slice(2)}</li>`;
      if (!l.trim()) return "";
      return `<p>${l}</p>`;
    })
    .join("");
  return <div className="report" dangerouslySetInnerHTML={{ __html: html }} />;
}
