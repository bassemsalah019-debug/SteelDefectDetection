"""
app.py - Streamlit app for Steel Surface Defect Detection (YOLOv8 / YOLO11 + XAI).

Run:
    streamlit run src/app.py

Features:
  - Upload an image OR capture from the webcam
  - Pick which trained model to use (defaults to the best: improved_opt)
  - Confidence threshold + inference size
  - Detections table (class + confidence)
  - Eigen-CAM explainability overlay ("where the model looks")

Fixes vs the original demo: points at a model that actually exists, and calls
register() before loading so the custom Ghost+MPCA+SIoU checkpoint un-pickles.
"""
import sys
from pathlib import Path

import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # make 'src' importable when run via `streamlit run`

from src.explain import EigenCAM, attention_summary, enable_custom_modules, overlay_cam  # noqa: E402
from src.infer import predict as run_detection  # noqa: E402
from src.preprocessing import channels_equal, to_model_input  # noqa: E402
from src.report import Detection, generate_report, to_html, to_pdf_bytes  # noqa: E402

CLASS_NAMES = [
    "crazing", "inclusion", "patches",
    "pitted_surface", "rolled-in_scale", "scratches",
]

# Friendly label -> weights path (relative to project root). Only existing ones are shown.
# Production model first (default). Per the 2026-06 audit + 5-seed study, the plain
# YOLOv8n baseline @640 is the best deployable model (0.7525 TEST, 0.7475±0.016 over 5
# seeds, 157 FPS, no TTA). The "improved" architectures did NOT beat it (docs/audit/).
MODEL_CHOICES = {
    "YOLOv8n baseline @640 - PRODUCTION (mAP@0.5 0.7525; 5-seed 0.7475±0.016; 157 FPS)": "results/baseline_640/weights/best.pt",
    "YOLOv8n baseline (opt @800+TTA, mAP@0.5 0.763)": "results/baseline_opt/weights/best.pt",
    "Improved Ghost+MPCA+SIoU (0.768 @800+TTA; not better than baseline - see audit)": "results/improved_opt/weights/best.pt",
    "LZY Ghost+ResCBAM+WIoU @640 (0.732)": "results/lzy_640/weights/best.pt",
    "YOLO11s @960 (0.746)": "results/yolo11s_960/weights/best.pt",
}

st.set_page_config(page_title="Steel Defect Detection", page_icon="🔩", layout="wide")
st.title("🔩 Steel Surface Defect Detection")
st.caption("YOLOv8 / YOLO11 on NEU-DET - 6 defect classes, with Eigen-CAM explainability")


@st.cache_resource(show_spinner="Loading model...")
def get_model(weights_path: str):
    from ultralytics import YOLO

    enable_custom_modules()  # so improved / LZY checkpoints un-pickle
    p = ROOT / weights_path
    return YOLO(str(p)) if p.exists() else None


@st.cache_resource(show_spinner="Preparing Eigen-CAM...")
def get_cam(weights_path: str):
    model = get_model(weights_path)
    return EigenCAM(model, device="cpu") if model is not None else None


available = {k: v for k, v in MODEL_CHOICES.items() if (ROOT / v).exists()}

with st.sidebar:
    st.header("Settings")
    if available:
        choice = st.selectbox("Model", list(available.keys()))
        weights = available[choice]
    else:
        weights = st.text_input(
            "Model weights (relative to project root)",
            "results/improved_opt/weights/best.pt",
        )
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)
    imgsz = st.select_slider("Inference size", options=[512, 640, 800], value=640)
    show_cam = st.checkbox("Show Eigen-CAM explanation", value=True)
    st.markdown("**Classes:** " + ", ".join(CLASS_NAMES))

model = get_model(weights)
if model is None:
    st.warning(
        f"No trained weights found at `{weights}`.\n\n"
        "Train a model first (notebooks 03 / updated_03 / updated_05), then select "
        "its `results/<run>/weights/best.pt` here."
    )
    st.stop()

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

    # Canonical preprocessing: collapse to grayscale-replicated (what training
    # saw) BEFORE detection. Fixes the RGB->grayscale mismatch at the source.
    model_input = to_model_input(image)
    result = run_detection(model, image, conf=conf, imgsz=imgsz, verbose=False)
    annotated = result.plot()[:, :, ::-1]  # BGR -> RGB (drawn on the gray input)
    cols[1].subheader("Detections")
    cols[1].image(annotated, use_container_width=True)
    if channels_equal(model_input):
        cols[1].caption("✓ Input collapsed to grayscale (NEU-DET parity) before detection.")

    if show_cam:
        cam = get_cam(weights)
        heat = cam(image, imgsz=imgsz)
        cols[2].subheader("Eigen-CAM")
        cols[2].image(overlay_cam(image, heat), use_container_width=True)
        cols[2].caption("Bright = regions driving the detector. Diffuse maps on "
                        "crazing / rolled-in_scale explain their lower accuracy.")

    st.subheader("Detected defects")
    if len(result.boxes) == 0:
        st.info("No defects detected above the confidence threshold.")
    else:
        rows = [
            {"Defect": CLASS_NAMES[int(b.cls[0])], "Confidence": f"{float(b.conf[0]):.1%}"}
            for b in result.boxes
        ]
        st.table(rows)

    # ---- Bilingual LLM defect report (grounded in the knowledge base) ----
    st.divider()
    st.subheader("📝 Defect report")
    lang_label = st.radio("Report language", ["English", "العربية (Arabic)"],
                          horizontal=True, key="report_lang")
    lang_code = "ar" if lang_label.startswith("الع") else "en"
    if st.button("Generate report", type="primary"):
        dets = [
            Detection(CLASS_NAMES[int(b.cls[0])], float(b.conf[0]),
                      [float(v) for v in b.xyxy[0].tolist()])
            for b in result.boxes
        ]
        xai_txt = attention_summary(heat, dets, img_size=image.size) if show_cam else None
        meta = {
            "source": source,
            "filename": getattr(up, "name", None) if source == "Upload image" else "webcam",
            "image_size": list(image.size),
            "model": choice if available else weights,
            "conf_threshold": conf,
            "imgsz": imgsz,
        }
        with st.spinner("Generating report (LLM endpoint if configured, else KB fallback)..."):
            rep = generate_report(dets, lang=lang_code, xai_summary=xai_txt, image_meta=meta)
        if rep["used_llm"]:
            st.success("Generated with the configured LLM endpoint.")
        else:
            st.info("LLM endpoint not reachable — generated a grounded knowledge-base report "
                    "(fallback). Set STEEL_LLM_BASE_URL / STEEL_LLM_MODEL and serve MiMo to "
                    "enable the LLM.")
        st.markdown(rep["text"])
        c1, c2 = st.columns(2)
        c1.download_button("⬇ Download HTML", to_html(rep).encode("utf-8"),
                           file_name=f"steel_report_{lang_code}.html", mime="text/html")
        try:
            c2.download_button("⬇ Download PDF", to_pdf_bytes(rep),
                               file_name=f"steel_report_{lang_code}.pdf", mime="application/pdf")
        except Exception as exc:
            c2.caption(f"PDF export unavailable: {exc}")
