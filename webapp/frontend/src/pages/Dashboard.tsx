import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type DashboardStats } from "../api";
import { useI18n } from "../i18n";
import { ClassBars, Spinner, Tile, fmtDate } from "../components";

export default function Dashboard() {
  const { t } = useI18n();
  const [s, setS] = useState<DashboardStats | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => { api.dashboard().then(setS).catch((e) => setErr(e.message)); }, []);

  if (err) return <div className="alert err">{err}</div>;
  if (!s) return <div className="center"><Spinner /></div>;

  const maxDay = Math.max(1, ...s.over_time.map((d) => d.defects));

  return (
    <div>
      <div className="topbar">
        <div>
          <div className="page-title">{t("dashboard")}</div>
          <div className="page-sub">{t("tagline")}</div>
        </div>
        <Link to="/new" className="btn">＋ {t("new_inspection")}</Link>
      </div>

      <div className="kpis" style={{ marginBottom: 18 }}>
        <Tile label={t("inspections")} value={s.total_inspections} />
        <Tile label={t("images")} value={s.total_images} />
        <Tile label={t("defects")} value={s.total_defects} />
        <Tile label={t("avg_per_image")} value={s.avg_defects_per_image} />
      </div>

      <div className="grid2">
        <div className="card fade-up"><h3>{t("defects_by_class")}</h3><ClassBars counts={s.class_counts} /></div>
        <div className="card fade-up d1">
          <h3>{t("activity_14d")}</h3>
          <div className="chart">
            {s.over_time.map((d) => (
              <div className="col" key={d.date} title={`${d.date}: ${d.defects} ${t("defects")}`}>
                <div className="b" style={{ height: `${Math.round((d.defects / maxDay) * 100)}%` }} />
              </div>
            ))}
          </div>
          <div className="row" style={{ justifyContent: "space-between", marginTop: 8 }}>
            <span className="muted mono" style={{ fontSize: ".68rem" }}>{s.over_time[0]?.date}</span>
            <span className="muted mono" style={{ fontSize: ".68rem" }}>{s.over_time[s.over_time.length - 1]?.date}</span>
          </div>
        </div>
      </div>

      <div className="card fade-up d2" style={{ marginTop: 18 }}>
        <h3>{t("recent")}</h3>
        {s.recent.length ? (
          <div className="list">
            {s.recent.map((r) => (
              <Link to={`/inspection/${r.id}`} key={r.id} className="row-card">
                <span className={`badge ${r.mode}`}>{r.mode}</span>
                <div className="grow">
                  <div className="t">{r.title}</div>
                  <div className="m">{fmtDate(r.created_at)} · {r.n_images} {t("images")} · {r.n_defects} {t("defects")}</div>
                </div>
                <span className="muted">→</span>
              </Link>
            ))}
          </div>
        ) : (
          <div className="muted">{t("no_inspections")} — {t("start_first")}</div>
        )}
      </div>
    </div>
  );
}
