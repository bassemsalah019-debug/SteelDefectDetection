import { useState, type FormEvent } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";
import { useI18n } from "../i18n";
import { Spinner } from "../components";

export default function Login() {
  const { login, signup } = useAuth();
  const { t } = useI18n();
  const navigate = useNavigate();
  const [mode, setMode] = useState<"login" | "signup">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setErr(""); setBusy(true);
    try {
      if (mode === "login") await login(email, password);
      else await signup(email, password, name);
      navigate("/");
    } catch (ex) {
      setErr((ex as Error).message || "Failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app-bg auth-wrap">
      <div className="auth-card fade-up">
        <div className="auth-hero">
          <div className="logo">🔩</div>
          <h2 style={{ margin: "0 0 4px" }}>{t("brand")}</h2>
          <div className="muted">{t("tagline")}</div>
        </div>
        <div className="auth-tabs">
          <button className={mode === "login" ? "on" : ""} onClick={() => setMode("login")}>{t("login")}</button>
          <button className={mode === "signup" ? "on" : ""} onClick={() => setMode("signup")}>{t("signup")}</button>
        </div>
        <form onSubmit={submit}>
          {mode === "signup" && (
            <div className="field">
              <label>{t("full_name")}</label>
              <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Inspector" />
            </div>
          )}
          <div className="field">
            <label>{t("email")}</label>
            <input type="email" required value={email} onChange={(e) => setEmail(e.target.value)} placeholder="me@steel.io" />
          </div>
          <div className="field">
            <label>{t("password")}</label>
            <input type="password" required minLength={8} value={password} onChange={(e) => setPassword(e.target.value)} placeholder="••••••••" />
          </div>
          {err && <div className="alert err">{err}</div>}
          <button className="btn" style={{ width: "100%", marginTop: 6 }} disabled={busy}>
            {busy ? <Spinner /> : mode === "login" ? t("welcome_back") : t("create_account")}
          </button>
        </form>
      </div>
    </div>
  );
}
