"""
app.py - Hugging Face Spaces entrypoint for the Steel Surface Defect Detection STUDIO.

Public CPU demo mirroring src/app.py with the same "Molten Graphite" design system:
  - grayscale-parity preprocessing · YOLOv8n baseline (bundled weights/best.pt)
  - Fixed + Adaptive confidence thresholding (src/adaptive_threshold.py)
  - Eigen-CAM explainability · grounded bilingual (EN/AR) report (KB fallback on CPU)

Expects the Space to contain: src/ , weights/best.pt . See docs/DEPLOYMENT.md.
Locally run the full app:  streamlit run src/app.py
"""
import sys
import time
from pathlib import Path

import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
for cand in (HERE, HERE.parent.parent):
    if (cand / "src").exists():
        sys.path.insert(0, str(cand))
        ROOT = cand
        break
else:
    ROOT = HERE

from src.adaptive_threshold import CLASS_NAMES as _AT_CLASSES  # noqa: E402
from src.explain import EigenCAM, attention_summary, enable_custom_modules, overlay_cam  # noqa: E402
from src.infer import predict as run_detection, predict_adaptive  # noqa: E402
from src.preprocessing import channels_equal, to_model_input  # noqa: E402

try:
    from src.report import Detection, generate_report, to_html, to_pdf_bytes
    HAS_REPORT = True
except Exception:
    HAS_REPORT = False

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]
CLASS_COLORS = {
    "crazing": "#ff5d5d", "inclusion": "#ff9f43", "patches": "#feca57",
    "pitted_surface": "#1dd1a1", "rolled-in_scale": "#54a0ff", "scratches": "#c56cf0",
}
WEIGHT_CANDIDATES = [ROOT / "weights" / "best.pt",
                     ROOT / "results" / "baseline_640" / "weights" / "best.pt",
                     HERE / "weights" / "best.pt"]

st.set_page_config(page_title="Steel Defect Detection Studio", page_icon="🔩", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@500;700&display=swap');
:root{ --panel:rgba(255,255,255,.04); --line:rgba(255,255,255,.09); --muted:#8b98a9;
  --accent:#ff6b35; --accent2:#ff3b6b; --mint:#2ee6a6; --font:'Space Grotesk',system-ui,sans-serif; --mono:'JetBrains Mono',monospace; }
html, body, [class*="css"]{ font-family:var(--font); }
.stApp{ background:radial-gradient(900px 500px at 88% -8%, rgba(255,107,53,.16), transparent 60%),
  radial-gradient(800px 500px at -8% 8%, rgba(84,160,255,.12), transparent 55%),
  linear-gradient(180deg,#0a0e14,#0b0f17); background-attachment:fixed; }
.block-container{ padding-top:1.2rem; max-width:1340px; }
#MainMenu, footer{ visibility:hidden; } [data-testid="stHeader"]{ background:transparent; }
[data-testid="stImage"] img{ border-radius:14px; border:1px solid var(--line); box-shadow:0 10px 28px rgba(0,0,0,.45); }
.hero{ position:relative; overflow:hidden; border-radius:24px; padding:30px 36px; margin-bottom:14px;
  border:1px solid var(--line); background:linear-gradient(125deg,#0f1722,#10141d 60%,#1a1320); }
.hero::before{ content:""; position:absolute; inset:-40%; animation:spin 18s linear infinite; filter:blur(28px); opacity:.55;
  background:conic-gradient(from 0deg, rgba(255,107,53,0), rgba(255,107,53,.18), rgba(255,59,107,.16), rgba(84,160,255,.12), rgba(255,107,53,0)); }
.hero>*{ position:relative; z-index:1; } @keyframes spin{ to{ transform:rotate(360deg); } }
.brand{ display:flex; align-items:center; gap:14px; }
.brand .logo{ width:46px; height:46px; border-radius:13px; display:grid; place-items:center; font-size:1.5rem;
  background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 8px 22px rgba(255,59,107,.4); }
.brand h1{ margin:0; font-size:1.55rem; font-weight:700; color:#fff; } .brand .sub{ color:var(--muted); font-size:.88rem; margin-top:2px; }
.pills{ margin-top:14px; display:flex; flex-wrap:wrap; gap:8px; }
.pill{ font-family:var(--mono); font-size:.72rem; color:#d8e0ea; padding:4px 11px; border-radius:999px; background:rgba(255,255,255,.05); border:1px solid var(--line); }
.led{ display:inline-flex; align-items:center; gap:7px; font-family:var(--mono); font-size:.72rem; color:var(--mint);
  padding:4px 11px; border-radius:999px; background:rgba(46,230,166,.08); border:1px solid rgba(46,230,166,.3); }
.led i{ width:8px; height:8px; border-radius:50%; background:var(--mint); box-shadow:0 0 10px var(--mint); animation:pulse 1.6s ease-in-out infinite; }
@keyframes pulse{ 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.45;transform:scale(.7)} }
.kick{ display:flex; align-items:center; gap:11px; margin:26px 0 12px; }
.kick .b{ width:26px; height:3px; border-radius:3px; background:linear-gradient(90deg,var(--accent),var(--accent2)); }
.kick .t{ font-weight:600; font-size:1.14rem; color:#fff; }
.kick .n{ font-family:var(--mono); font-size:.68rem; color:var(--accent); border:1px solid var(--line); border-radius:6px; padding:1px 7px; }
.glass{ background:var(--panel); border:1px solid var(--line); border-radius:18px; padding:18px 20px; backdrop-filter:blur(10px); }
.kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:13px; }
.tile{ position:relative; overflow:hidden; background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:15px 17px; backdrop-filter:blur(10px); transition:transform .2s, border-color .2s; }
.tile:hover{ transform:translateY(-3px); border-color:rgba(255,107,53,.55); }
.tile::after{ content:""; position:absolute; right:-18px; top:-18px; width:64px; height:64px; background:radial-gradient(circle, rgba(255,107,53,.28), transparent 70%); }
.tile .lab{ font-size:.66rem; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); }
.tile .val{ font-family:var(--mono); font-weight:700; font-size:1.55rem; color:#fff; margin-top:5px; line-height:1; }
.tile .sub{ font-family:var(--mono); font-size:.7rem; color:var(--accent); margin-top:5px; }
.cls-row{ display:flex; align-items:center; gap:12px; margin:9px 0; }
.cls-name{ width:138px; font-size:.82rem; font-weight:600; color:#dfe7ef; display:flex; align-items:center; gap:8px; }
.bar-wrap{ flex:1; height:18px; border-radius:10px; overflow:hidden; background:rgba(255,255,255,.06); border:1px solid var(--line); }
.bar{ height:100%; border-radius:10px; background-image:linear-gradient(90deg, rgba(255,255,255,.30), rgba(255,255,255,0)); }
.cls-val{ width:92px; text-align:right; font-family:var(--mono); font-size:.74rem; color:var(--muted); }
.legend{ display:flex; flex-wrap:wrap; gap:7px; }
.chip{ display:inline-flex; align-items:center; gap:7px; font-size:.76rem; color:#cdd7e1; background:rgba(255,255,255,.04); border:1px solid var(--line); border-radius:999px; padding:3px 10px; }
.dot{ width:9px; height:9px; border-radius:50%; display:inline-block; }
.parity{ background:rgba(46,230,166,.09); color:var(--mint); border:1px solid rgba(46,230,166,.3); border-radius:10px; padding:7px 11px; font-size:.8rem; margin-top:9px; }
.imlabel{ font-size:.7rem; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
[data-testid="stSidebar"]{ background:linear-gradient(180deg,#0c111a,#0a0e14); border-right:1px solid var(--line); }
.stButton>button{ border:none; border-radius:12px; font-weight:600; color:#fff; background:linear-gradient(135deg,var(--accent),var(--accent2)); box-shadow:0 8px 20px rgba(255,59,107,.32); transition:transform .15s, box-shadow .15s; }
.stButton>button:hover{ transform:translateY(-2px); box-shadow:0 12px 28px rgba(255,59,107,.5); }
[data-testid="stDownloadButton"]>button{ background:transparent; border:1px solid var(--line); color:#dbe3ec; box-shadow:none; font-weight:500; }
.stTabs [data-baseweb="tab-list"]{ background:var(--panel); border:1px solid var(--line); border-radius:13px; padding:5px; gap:4px; }
.stTabs [data-baseweb="tab"]{ border-radius:9px; color:var(--muted); padding:6px 16px; }
.stTabs [aria-selected="true"]{ background:linear-gradient(135deg,var(--accent),var(--accent2))!important; color:#fff!important; }
.foot{ text-align:center; color:var(--muted); font-size:.78rem; margin-top:30px; border-top:1px solid var(--line); padding-top:14px; }
</style>
""",
    unsafe_allow_html=True,
)


def kick(title, n):
    return f'<div class="kick"><span class="b"></span><span class="t">{title}</span><span class="n">{n}</span></div>'


def kpi_tiles(items):
    cells = []
    for it in items:
        sub = f'<div class="sub">{it["sub"]}</div>' if it.get("sub") else ""
        cells.append(f'<div class="tile"><div class="lab">{it["label"]}</div>'
                     f'<div class="val">{it["value"]}</div>{sub}</div>')
    return f'<div class="kpis">{"".join(cells)}</div>'


def legend_html():
    return '<div class="legend">' + "".join(
        f'<span class="chip"><span class="dot" style="background:{CLASS_COLORS[c]};box-shadow:0 0 8px {CLASS_COLORS[c]}"></span>{c}</span>'
        for c in CLASS_NAMES) + "</div>"


def per_class_bars(rows):
    out = []
    for name, count, mconf in rows:
        col = CLASS_COLORS.get(name, "#888"); pct = max(3, int(round(mconf * 100)))
        out.append(f'<div class="cls-row"><div class="cls-name"><span class="dot" style="background:{col};box-shadow:0 0 8px {col}"></span>{name}</div>'
                   f'<div class="bar-wrap"><div class="bar" style="width:{pct}%;background-color:{col};box-shadow:0 0 14px {col}88"></div></div>'
                   f'<div class="cls-val">{count}× · {int(round(mconf*100))}%</div></div>')
    return "".join(out)


st.markdown(
    """
<div class="hero">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
    <div class="brand"><div class="logo">🔩</div>
      <div><h1>Steel Defect Detection Studio</h1>
      <div class="sub">YOLOv8n · Adaptive thresholding · Eigen-CAM · bilingual reports — public CPU demo</div></div></div>
    <span class="led"><i></i> SYSTEM LIVE</span>
  </div>
  <div class="pills">
    <span class="pill">6 defect classes</span><span class="pill">TEST mAP@0.5 · 0.7525</span>
    <span class="pill">Grayscale-parity preprocessing</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading model…")
def load():
    from ultralytics import YOLO

    enable_custom_modules()
    for w in WEIGHT_CANDIDATES:
        if w.exists():
            model = YOLO(str(w))
            return model, EigenCAM(model, device="cpu")
    return None, None


model, cam = load()
if model is None:
    st.error("Model weight not found. Add `weights/best.pt` to the Space (see docs/DEPLOYMENT.md).")
    st.stop()

with st.sidebar:
    st.markdown("### ⚙️ Control panel")
    threshold_mode = st.radio("🎯 Thresholding mode", ["Adaptive", "Fixed"], horizontal=True,
                              help="Adaptive = per-class, per-image confidence (post-processing only). Fixed = one global conf.")
    conf = st.slider("Fixed confidence", 0.0, 1.0, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", options=[512, 640, 800], value=640)
    show_cam = st.toggle("Eigen-CAM overlay", value=True)
    if HAS_REPORT:
        lang_label = st.radio("Report language", ["English", "العربية (Arabic)"], horizontal=True)
        lang_code = "ar" if lang_label.startswith("الع") else "en"
    st.markdown("#### 🏷️ Defect classes")
    st.markdown(legend_html(), unsafe_allow_html=True)
    st.caption("Public CPU demo. Reports use the grounded knowledge base (no live LLM here).")

st.markdown(kick("Input", "01"), unsafe_allow_html=True)
tab_up, tab_cam = st.tabs(["📤  Upload image", "📷  Webcam"])
with tab_up:
    up = st.file_uploader("Drop a steel-surface image", type=["jpg", "jpeg", "png", "bmp"], label_visibility="collapsed")
with tab_cam:
    shot = st.camera_input("Capture a steel surface", label_visibility="collapsed")

image = None
if up is not None:
    image = Image.open(up).convert("RGB")
elif shot is not None:
    image = Image.open(shot).convert("RGB")

if image is None:
    st.markdown('<div class="glass" style="margin-top:6px"><div style="font-size:1.05rem;font-weight:600;color:#fff;margin-bottom:6px">👋 Get started</div>'
                '<div style="color:#b9c4d0;line-height:1.6">Upload a steel-surface image (or use the webcam). The model collapses it to '
                '<b style="color:#fff">grayscale</b> (as trained), draws boxes, shows <b style="color:#fff">where it looked</b> with Eigen-CAM, '
                'and generates a downloadable <b style="color:#fff">bilingual report</b>. Try <b style="color:var(--accent)">Adaptive</b> mode for '
                f'per-class, per-image thresholds.</div><div style="margin-top:14px">{legend_html()}</div></div>', unsafe_allow_html=True)
    st.stop()

model_input = to_model_input(image)
adaptive = None
t0 = time.perf_counter()
if threshold_mode == "Adaptive":
    adaptive = predict_adaptive(model, image, imgsz=imgsz, verbose=False)
    result = adaptive.result
else:
    result = run_detection(model, image, conf=conf, imgsz=imgsz, verbose=False)
latency_ms = (time.perf_counter() - t0) * 1000.0

boxes = result.boxes
n_det = 0 if boxes is None else len(boxes)
confs = [float(b.conf[0]) for b in boxes] if n_det else []
classes = [CLASS_NAMES[int(b.cls[0])] for b in boxes] if n_det else []
mean_conf = sum(confs) / len(confs) if confs else 0.0

st.markdown(kick("Results", "02"), unsafe_allow_html=True)
st.markdown(kpi_tiles([
    {"label": "Defects found", "value": n_det},
    {"label": "Distinct classes", "value": len(set(classes))},
    {"label": "Mean confidence", "value": f"{mean_conf:.0%}" if confs else "—"},
    {"label": "Latency", "value": f"{latency_ms:.0f} ms"},
    {"label": "Mode", "value": "⚡ Adaptive" if adaptive is not None else "🔒 Fixed"},
]), unsafe_allow_html=True)

st.write("")
cols = st.columns(3 if show_cam else 2)
cols[0].markdown('<div class="imlabel">Input · grayscale parity</div>', unsafe_allow_html=True)
cols[0].image(model_input, width="stretch")
if channels_equal(model_input):
    cols[0].markdown('<div class="parity">✓ Collapsed to grayscale (R=G=B) before detection.</div>', unsafe_allow_html=True)
cols[1].markdown('<div class="imlabel">Detections</div>', unsafe_allow_html=True)
cols[1].image(result.plot()[:, :, ::-1], width="stretch")

heat = None
if show_cam:
    heat = cam(image, imgsz=imgsz)
    cols[2].markdown('<div class="imlabel">Eigen-CAM</div>', unsafe_allow_html=True)
    cols[2].image(overlay_cam(image, heat), width="stretch")

st.write("")
left, right = st.columns([3, 2])
with left:
    st.markdown('<div class="imlabel">Detected defects</div>', unsafe_allow_html=True)
    if n_det == 0:
        st.info("No defects detected above the current threshold(s).")
    else:
        try:
            import pandas as pd

            df = pd.DataFrame({"Defect": classes, "Confidence": [c * 100 for c in confs]})
            st.dataframe(df, hide_index=True, width="stretch", column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", format="%.1f%%", min_value=0, max_value=100)})
        except Exception:
            st.table([{"Defect": c, "Confidence": f"{v:.1%}"} for c, v in zip(classes, confs)])
with right:
    st.markdown('<div class="imlabel">Per-class summary</div>', unsafe_allow_html=True)
    if n_det:
        agg = {}
        for c, v in zip(classes, confs):
            agg.setdefault(c, []).append(v)
        rows = [(c, len(vs), sum(vs) / len(vs)) for c, vs in sorted(agg.items(), key=lambda kv: -len(kv[1]))]
        st.markdown('<div class="glass">' + per_class_bars(rows) + '</div>', unsafe_allow_html=True)
    else:
        st.caption("—")

if adaptive is not None:
    st.markdown(kick("Adaptive thresholding", "03"), unsafe_allow_html=True)
    sig, thr = adaptive.detail["signals"], adaptive.detail["thresholds"]
    st.markdown(kpi_tiles([
        {"label": "Brightness", "value": f"{sig['brightness']:.2f}"},
        {"label": "Image quality", "value": f"{sig['quality']:.2f}"},
        {"label": "Sharpness", "value": f"{sig['sharpness']:.0f}"},
        {"label": "Candidates", "value": sig["density"]},
    ]), unsafe_allow_html=True)
    st.write("")
    try:
        import pandas as pd

        tdf = pd.DataFrame({"Defect": _AT_CLASSES, "Adaptive conf": [thr[c] for c in _AT_CLASSES]})
        st.dataframe(tdf, hide_index=True, width="stretch", column_config={
            "Adaptive conf": st.column_config.ProgressColumn("Adaptive conf", format="%.3f", min_value=0.0, max_value=0.6)})
    except Exception:
        st.table([{"Defect": c, "Adaptive conf": f"{thr[c]:.3f}"} for c in _AT_CLASSES])

if HAS_REPORT:
    st.markdown(kick("Inspection report", "✎"), unsafe_allow_html=True)
    if st.button("Generate report", type="primary"):
        dets = [Detection(CLASS_NAMES[int(b.cls[0])], float(b.conf[0]),
                          [float(v) for v in b.xyxy[0].tolist()]) for b in result.boxes]
        xai_txt = attention_summary(heat, dets, img_size=image.size) if heat is not None else None
        meta = {"image_size": list(image.size), "imgsz": imgsz, "threshold_mode": threshold_mode}
        if adaptive is not None:
            meta["adaptive_thresholds"] = adaptive.detail["thresholds"]
        with st.spinner("Generating grounded report…"):
            rep = generate_report(dets, lang=lang_code, xai_summary=xai_txt, image_meta=meta)
        st.caption("Knowledge-base fallback (no LLM endpoint on this demo)." if not rep["used_llm"]
                   else "Generated with the configured LLM endpoint.")
        with st.container(border=True):
            st.markdown(rep["text"])
        c1, c2 = st.columns(2)
        c1.download_button("⬇ HTML", to_html(rep).encode("utf-8"), file_name=f"steel_report_{lang_code}.html",
                           mime="text/html", width="stretch")
        try:
            c2.download_button("⬇ PDF", to_pdf_bytes(rep), file_name=f"steel_report_{lang_code}.pdf",
                               mime="application/pdf", width="stretch")
        except Exception as exc:
            c2.caption(f"PDF unavailable: {exc}")

st.markdown('<div class="foot">Steel Surface Defect Detection · <b>YOLOv8n + Adaptive thresholding '
            '+ Eigen-CAM + bilingual reporting</b></div>', unsafe_allow_html=True)
