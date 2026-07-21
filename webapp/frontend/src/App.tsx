import { Navigate, NavLink, Outlet, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth";
import { useI18n } from "./i18n";
import { Spinner } from "./components";
import Login from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import NewInspection from "./pages/NewInspection";
import History from "./pages/History";
import Detail from "./pages/Detail";

function Shell() {
  const { user, logout } = useAuth();
  const { t, lang, setLang } = useI18n();
  const links = [
    { to: "/", icon: "▦", key: "dashboard", end: true },
    { to: "/new", icon: "＋", key: "new_inspection", end: false },
    { to: "/history", icon: "🕔", key: "history", end: false },
  ];
  return (
    <div className="app-bg">
      <div className="shell">
        <aside className="sidebar">
          <div className="brand">
            <div className="logo">🔩</div>
            <div><b>{t("brand")}</b><small>{t("tagline")}</small></div>
          </div>
          <nav className="nav">
            {links.map((n) => (
              <NavLink key={n.to} to={n.to} end={n.end} className={({ isActive }) => (isActive ? "active" : "")}>
                <span className="ico">{n.icon}</span>{t(n.key)}
              </NavLink>
            ))}
          </nav>
          <div className="spacer" />
          <div className="userbox">
            <div className="seg" style={{ marginBottom: 12 }}>
              <button className={lang === "en" ? "on" : ""} onClick={() => setLang("en")}>EN</button>
              <button className={lang === "ar" ? "on" : ""} onClick={() => setLang("ar")}>ع</button>
            </div>
            <div style={{ marginBottom: 8, color: "#cdd7e1", fontSize: ".88rem" }}>{user?.full_name || user?.email}</div>
            <button className="btn ghost sm" onClick={logout}>⏻ {t("logout")}</button>
          </div>
        </aside>
        <main className="main"><Outlet /></main>
      </div>
    </div>
  );
}

function Protected() {
  const { user, loading } = useAuth();
  if (loading) return <div className="app-bg center"><Spinner /></div>;
  if (!user) return <Navigate to="/login" replace />;
  return <Shell />;
}

export default function App() {
  const { user, loading } = useAuth();
  return (
    <Routes>
      <Route
        path="/login"
        element={loading ? <div className="app-bg center"><Spinner /></div> : user ? <Navigate to="/" replace /> : <Login />}
      />
      <Route element={<Protected />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/new" element={<NewInspection />} />
        <Route path="/history" element={<History />} />
        <Route path="/inspection/:id" element={<Detail />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
