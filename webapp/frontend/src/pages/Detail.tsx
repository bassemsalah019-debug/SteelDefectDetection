import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, type ImageOut, type InspectionDetail, type ReportOut } from "../api";
import { mediaUrl } from "../config";
import { useI18n } from "../i18n";
import { ClassBars, ReportText, Spinner, fmtDate } from "../components";

type View = "annotated" | "original" | "cam";

export default function Detail() {
  const { id } = useParams();
  const { t, lang } = useI18n();
  const [insp, setInsp] = useState<InspectionDetail | null>(null);
  const [err, setErr] = useState("");
  const [view, setView] = useState<View>("annotated");
  const [report, setReport] = useState<ReportOut | null>(null);
  const [genBusy, setGenBusy] = useState(false);

  useEffect(() => { if (id) api.getInspection(id).then(setInsp).catch((e) => setErr(e.message)); }, [id]);

  const gen = async () => {
    if (!id) return;
    setGenBusy(true);
    try { setReport(await api.report(id, lang)); }
    catch (e) { setErr((e as Error).message); }
    finally { setGenBusy(false); }
  };

  if (err) return <div className="alert err">{err}</div>;
  if (!insp) return <div className="center"><Spinner /></div>;

  const urlFor = (im: ImageOut) =>
    view === "original" ? im.original_url : view === "cam" ? im.cam_url || im.annotated_url : im.annotated_url;

  return (
    <div>
      <div className="topbar">
        <div className="row">
          <Link to="/history" className="btn ghost sm">← {t("back")}</Link>
          <div>
            <div className="page-title">{insp.title}</div>
            <div className="page-sub">{fmtDate(insp.created_at)} · {insp.n_images} {t("images")} · {insp.n_defects} {t("defects")}</div>
          </div>
        </div>
        <span className={`badge ${insp.mode}`}>{insp.mode}</span>
      </div>

      <div className="seg" style={{ marginBottom: 14 }}>
        <button className={view === "annotated" ? "on" : ""} onClick={() => setView("annotated")}>{t("detections")}</button>
        <button className={view === "cam" ? "on" : ""} onClick={() => setView("cam")}>{t("eigencam")}</button>
        <button className={view === "original" ? "on" : ""} onClick={() => setView("original")}>{t("original")}</button>
      </div>

      <div className="detgrid">
        {insp.images.map((im) => (
          <div className="card fade-up" key={im.id}>
            <img className="frame" src={mediaUrl(urlFor(im))} alt={im.filename} />
            <div className="row wrap" style={{ gap: 8, marginTop: 10, justifyContent: "space-between" }}>
              <span className="muted" style={{ fontSize: ".8rem" }}>{im.filename}</span>
              <span className="chip">{im.n_defects} {t("defects_found")}</span>
            </div>
            {im.brightness != null && (
              <div className="muted mono" style={{ fontSize: ".72rem", marginTop: 6 }}>
                {t("brightness")} {im.brightness?.toFixed(2)} · {t("quality")} {im.quality?.toFixed(2)}
              </div>
            )}
            {im.detections.length > 0 && (
              <table className="table" style={{ marginTop: 10 }}>
                <thead><tr><th>{t("detections")}</th><th className="right">{t("confidence")}</th></tr></thead>
                <tbody>
                  {im.detections.map((d) => (
                    <tr key={d.id}>
                      <td>{d.cls_name}</td>
                      <td className="right conf-pill">{(d.confidence * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        ))}
      </div>

      <div className="grid2" style={{ marginTop: 18 }}>
        <div className="card"><h3>{t("per_class")}</h3><ClassBars counts={insp.class_counts} /></div>
        <div className="card">
          <h3>{t("report")}</h3>
          <button className="btn" onClick={gen} disabled={genBusy}>
            {genBusy ? <><Spinner /> {t("generating")}</> : t("generate_report")}
          </button>
          {report && <div style={{ marginTop: 14 }}><ReportText text={report.text} /></div>}
        </div>
      </div>
    </div>
  );
}
