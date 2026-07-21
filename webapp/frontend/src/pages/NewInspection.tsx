import { useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useI18n } from "../i18n";
import { Spinner } from "../components";

export default function NewInspection() {
  const { t } = useI18n();
  const navigate = useNavigate();
  const [files, setFiles] = useState<File[]>([]);
  const [mode, setMode] = useState<"adaptive" | "fixed">("adaptive");
  const [conf, setConf] = useState(0.25);
  const [imgsz, setImgsz] = useState(640);
  const [title, setTitle] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const [over, setOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const addFiles = (fl: FileList | null) => {
    if (!fl) return;
    setFiles((prev) => [...prev, ...Array.from(fl)].slice(0, 20));
  };

  const submit = async () => {
    if (!files.length) { setErr("Add at least one image"); return; }
    setErr(""); setBusy(true);
    try {
      const fd = new FormData();
      files.forEach((f) => fd.append("files", f));
      fd.append("title", title || "Untitled inspection");
      fd.append("mode", mode);
      fd.append("conf", String(conf));
      fd.append("imgsz", String(imgsz));
      const res = await api.createInspection(fd);
      navigate(`/inspection/${res.id}`);
    } catch (ex) {
      setErr((ex as Error).message || "Failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div>
      <div className="topbar">
        <div>
          <div className="page-title">{t("new_inspection")}</div>
          <div className="page-sub">{t("drop_images")}</div>
        </div>
      </div>

      <div className="grid2">
        <div className="card">
          <div
            className={`drop ${over ? "over" : ""}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={(e) => { e.preventDefault(); setOver(true); }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => { e.preventDefault(); setOver(false); addFiles(e.dataTransfer.files); }}
          >
            <div style={{ fontSize: "2.2rem" }}>🖼️</div>
            <div style={{ marginTop: 8 }}>{t("drop_images")}</div>
            <input ref={inputRef} type="file" accept="image/*" multiple hidden onChange={(e) => addFiles(e.target.files)} />
          </div>
          {files.length > 0 && (
            <>
              <div className="thumbs">
                {files.map((f, i) => <img key={i} className="thumb" src={URL.createObjectURL(f)} alt={f.name} />)}
              </div>
              <div className="muted" style={{ marginTop: 8, fontSize: ".82rem" }}>{files.length} {t("selected")}</div>
            </>
          )}
        </div>

        <div className="card">
          <div className="field">
            <label>{t("title")}</label>
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Coil A-204 scan" />
          </div>
          <div className="field">
            <label>{t("mode")}</label>
            <div className="seg">
              <button className={mode === "adaptive" ? "on" : ""} onClick={() => setMode("adaptive")}>⚡ {t("adaptive")}</button>
              <button className={mode === "fixed" ? "on" : ""} onClick={() => setMode("fixed")}>🔒 {t("fixed")}</button>
            </div>
          </div>
          {mode === "fixed" && (
            <div className="field">
              <label>{t("confidence")}: {conf.toFixed(2)}</label>
              <input type="range" min={0} max={1} step={0.05} value={conf} onChange={(e) => setConf(Number(e.target.value))} />
            </div>
          )}
          <div className="field">
            <label>{t("inference_size")}</label>
            <select value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>
              <option value={512}>512</option>
              <option value={640}>640</option>
              <option value={800}>800</option>
            </select>
          </div>
          {err && <div className="alert err">{err}</div>}
          <button className="btn" style={{ width: "100%" }} disabled={busy || !files.length} onClick={submit}>
            {busy ? <><Spinner /> {t("analyzing")}</> : t("run_inspection")}
          </button>
        </div>
      </div>
    </div>
  );
}
