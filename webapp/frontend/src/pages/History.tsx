import { useEffect, useState, type MouseEvent } from "react";
import { Link } from "react-router-dom";
import { api, type Inspection } from "../api";
import { useI18n } from "../i18n";
import { Spinner, fmtDate } from "../components";

const PAGE_SIZE = 10;

export default function History() {
  const { t } = useI18n();
  const [items, setItems] = useState<Inspection[] | null>(null);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [mode, setMode] = useState("");
  const [err, setErr] = useState("");

  const load = () => {
    setItems(null);
    api.listInspections(page, PAGE_SIZE, mode || undefined)
      .then((p) => { setItems(p.items); setTotal(p.total); })
      .catch((e) => setErr(e.message));
  };
  useEffect(load, [page, mode]);

  const del = async (id: string, e: MouseEvent) => {
    e.preventDefault();
    if (!confirm("Delete this inspection?")) return;
    await api.deleteInspection(id);
    load();
  };

  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div>
      <div className="topbar">
        <div>
          <div className="page-title">{t("history")}</div>
          <div className="page-sub">{total} {t("inspections")}</div>
        </div>
        <div className="seg">
          <button className={mode === "" ? "on" : ""} onClick={() => { setMode(""); setPage(1); }}>{t("all_modes")}</button>
          <button className={mode === "adaptive" ? "on" : ""} onClick={() => { setMode("adaptive"); setPage(1); }}>{t("adaptive")}</button>
          <button className={mode === "fixed" ? "on" : ""} onClick={() => { setMode("fixed"); setPage(1); }}>{t("fixed")}</button>
        </div>
      </div>

      {err && <div className="alert err">{err}</div>}

      {!items ? (
        <div className="center"><Spinner /></div>
      ) : items.length === 0 ? (
        <div className="card"><div className="muted">{t("no_inspections")} — {t("start_first")}</div></div>
      ) : (
        <div className="list">
          {items.map((r) => (
            <Link to={`/inspection/${r.id}`} key={r.id} className="row-card">
              <span className={`badge ${r.mode}`}>{r.mode}</span>
              <div className="grow">
                <div className="t">{r.title}</div>
                <div className="m">{fmtDate(r.created_at)} · {r.n_images} {t("images")} · {r.n_defects} {t("defects")}</div>
              </div>
              <button className="btn danger sm" onClick={(e) => del(r.id, e)}>{t("delete")}</button>
              <span className="muted">→</span>
            </Link>
          ))}
        </div>
      )}

      {pages > 1 && (
        <div className="row" style={{ justifyContent: "center", gap: 14, marginTop: 16 }}>
          <button className="btn ghost sm" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>←</button>
          <span className="muted mono">{page} / {pages}</span>
          <button className="btn ghost sm" disabled={page >= pages} onClick={() => setPage((p) => p + 1)}>→</button>
        </div>
      )}
    </div>
  );
}
