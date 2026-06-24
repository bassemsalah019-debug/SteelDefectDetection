"""
app.py - Hugging Face Spaces entrypoint for Steel Surface Defect Detection.

Mirrors the production app (src/app.py) for the public CPU demo:
  - grayscale-parity preprocessing (the RGB->grayscale fix) via src.preprocessing
  - the validated production model (YOLOv8n baseline @640), bundled as weights/best.pt
  - Eigen-CAM explainability
  - bilingual (EN/AR) defect report — on the free CPU Space there is no MiMo endpoint,
    so it gracefully degrades to the grounded knowledge-base report, with HTML/PDF download

Expects the Space to contain: src/ , weights/best.pt . See docs/DEPLOYMENT.md.
Locally you can run the full app instead:  streamlit run src/app.py
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

HERE = Path(__file__).resolve().parent
# Make 'src' importable whether the Space is flat or mirrors the repo.
for cand in (HERE, HERE.parent.parent):
    if (cand / "src").exists():
        sys.path.insert(0, str(cand))
        ROOT = cand
        break
else:
    ROOT = HERE

from src.explain import EigenCAM, attention_summary, enable_custom_modules, overlay_cam  # noqa: E402
from src.infer import predict as run_detection  # noqa: E402
from src.preprocessing import channels_equal, to_model_input  # noqa: E402

try:  # the report stack is optional; the core demo must work without it
    from src.report import Detection, generate_report, to_html, to_pdf_bytes
    HAS_REPORT = True
except Exception:
    HAS_REPORT = False

CLASS_NAMES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

WEIGHT_CANDIDATES = [
    ROOT / "weights" / "best.pt",
    ROOT / "results" / "baseline_640" / "weights" / "best.pt",
    HERE / "weights" / "best.pt",
]

st.set_page_config(page_title="Steel Defect Detection", page_icon="🔩", layout="wide")
st.title("🔩 Steel Surface Defect Detection")
st.caption("YOLOv8n on NEU-DET - 6 defect classes, Eigen-CAM explainability, bilingual report")


@st.cache_resource(show_spinner="Loading model...")
def load():
    from ultralytics import YOLO

    enable_custom_modules()  # harmless for the stock baseline; needed if a custom ckpt is bundled
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
    st.header("Settings")
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", options=[512, 640, 800], value=640)
    show_cam = st.checkbox("Show Eigen-CAM explanation", value=True)
    st.markdown("**Classes:** " + ", ".join(CLASS_NAMES))
    st.caption("Public CPU demo. Reports use the grounded knowledge base (no live LLM here).")

source = st.radio("Input source", ["Upload image", "Webcam"], horizontal=True)
if source == "Upload image":
    up = st.file_uploader("Upload a steel surface image", type=["jpg", "jpeg", "png", "bmp"])
    image = Image.open(up).convert("RGB") if up is not None else None
else:
    shot = st.camera_input("Capture a steel surface")
    image = Image.open(shot).convert("RGB") if shot is not None else None

if image is not None:
    cols = st.columns(3 if show_cam else 2)
    cols[0].subheader("Input")
    cols[0].image(image, use_container_width=True)

    # Canonical preprocessing (grayscale-replicate) BEFORE detection - fixes the RGB bug.
    model_input = to_model_input(image)
    result = run_detection(model, image, conf=conf, imgsz=imgsz, verbose=False)
    cols[1].subheader("Detections")
    cols[1].image(result.plot()[:, :, ::-1], use_container_width=True)
    if channels_equal(model_input):
        cols[1].caption("✓ Input collapsed to grayscale (NEU-DET parity) before detection.")

    heat = None
    if show_cam:
        heat = cam(image, imgsz=imgsz)
        cols[2].subheader("Eigen-CAM")
        cols[2].image(overlay_cam(image, heat), use_container_width=True)

    st.subheader("Detected defects")
    if len(result.boxes) == 0:
        st.info("No defects detected above the confidence threshold.")
    else:
        st.table([
            {"Defect": CLASS_NAMES[int(b.cls[0])], "Confidence": f"{float(b.conf[0]):.1%}"}
            for b in result.boxes
        ])

    if HAS_REPORT:
        st.divider()
        st.subheader("📝 Defect report")
        lang_label = st.radio("Report language", ["English", "العربية (Arabic)"],
                              horizontal=True, key="report_lang")
        lang_code = "ar" if lang_label.startswith("الع") else "en"
        if st.button("Generate report", type="primary"):
            dets = [Detection(CLASS_NAMES[int(b.cls[0])], float(b.conf[0]),
                              [float(v) for v in b.xyxy[0].tolist()]) for b in result.boxes]
            xai_txt = attention_summary(heat, dets, img_size=image.size) if heat is not None else None
            with st.spinner("Generating grounded report..."):
                rep = generate_report(dets, lang=lang_code, xai_summary=xai_txt,
                                      image_meta={"image_size": list(image.size), "imgsz": imgsz})
            st.caption("Generated with the knowledge-base fallback (no LLM endpoint on this demo)."
                       if not rep["used_llm"] else "Generated with the configured LLM endpoint.")
            st.markdown(rep["text"])
            c1, c2 = st.columns(2)
            c1.download_button("⬇ HTML", to_html(rep).encode("utf-8"),
                               file_name=f"steel_report_{lang_code}.html", mime="text/html")
            try:
                c2.download_button("⬇ PDF", to_pdf_bytes(rep),
                                   file_name=f"steel_report_{lang_code}.pdf", mime="application/pdf")
            except Exception as exc:
                c2.caption(f"PDF unavailable: {exc}")
