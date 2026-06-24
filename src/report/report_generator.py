"""
report_generator.py - bilingual (EN/AR) LLM defect report for the steel app.

Pipeline: detections (+ optional XAI attention summary + image metadata) are
combined with the grounded knowledge base (knowledge_base.json) into a structured
prompt, sent to an LLM through an OpenAI-compatible Chat Completions endpoint
(default: Xiaomi MiMo-7B-RL served by Ollama at http://localhost:11434/v1).

Design rules from the master prompt:
  - One thin, swappable provider interface: `generate(prompt, system) -> str`.
  - base_url / model / api_key come from ENVIRONMENT VARIABLES, never hardcoded.
  - The LLM is grounded in the KB and told to explain ONLY detected defects.
  - Graceful degradation: if the endpoint is unreachable, fall back to a
    deterministic, KB-only template report (clearly marked) so the app never breaks.
  - Export to HTML and PDF (Arabic shaped RTL via arabic_reshaper + python-bidi).

Environment variables (all optional - sensible local defaults):
    STEEL_LLM_BASE_URL   default "http://localhost:11434/v1"   (Ollama OpenAI API)
    STEEL_LLM_MODEL      default "mimo-7b-rl"
    STEEL_LLM_API_KEY    default "ollama"   (Ollama ignores it; any non-empty works)
"""
from __future__ import annotations

import html
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

_KB_PATH = Path(__file__).resolve().parent / "knowledge_base.json"

# Section headings per language (keeps the template + HTML consistent with the LLM).
SECTIONS = {
    "en": ["Executive Summary", "Per-defect Explanation", "XAI Interpretation",
           "Recommended Action", "Confidence Caveats"],
    "ar": ["الملخص التنفيذي", "شرح كل عيب", "تفسير الذكاء المُفسَّر (XAI)",
           "الإجراء المُوصى به", "تحفظات الثقة"],
}
LANG_NAME = {"en": "English", "ar": "Arabic"}


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Detection:
    """One YOLO detection. bbox is [x1, y1, x2, y2] in pixels (xyxy)."""
    cls_name: str
    confidence: float
    bbox: Sequence[float] = field(default_factory=list)


def load_kb(path: str | Path = _KB_PATH) -> dict:
    """Load the grounding knowledge base."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# Provider interface (swappable backend)
# --------------------------------------------------------------------------- #
class OpenAICompatProvider:
    """Talks to any OpenAI-compatible Chat Completions endpoint (MiMo via Ollama,
    vLLM, or a hosted base_url). Constructed lazily; no network call until generate()."""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float = 60.0):
        self.base_url, self.model, self.api_key, self.timeout = base_url, model, api_key, timeout
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            from openai import OpenAI  # imported lazily so the module loads without it

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=self.timeout)
        return self._client

    def generate(self, prompt: str, system: str | None = None) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        resp = self._ensure_client().chat.completions.create(
            model=self.model, messages=messages, temperature=0.2
        )
        return (resp.choices[0].message.content or "").strip()


def default_provider() -> OpenAICompatProvider:
    """Build the provider from environment variables (with local Ollama defaults)."""
    return OpenAICompatProvider(
        base_url=os.environ.get("STEEL_LLM_BASE_URL", "http://localhost:11434/v1"),
        model=os.environ.get("STEEL_LLM_MODEL", "mimo-7b-rl"),
        api_key=os.environ.get("STEEL_LLM_API_KEY", "ollama"),
    )


# --------------------------------------------------------------------------- #
# Prompt construction (grounded)
# --------------------------------------------------------------------------- #
def _detection_summary(detections: Sequence[Detection]) -> dict:
    counts: dict[str, int] = {}
    confs: dict[str, list[float]] = {}
    for d in detections:
        counts[d.cls_name] = counts.get(d.cls_name, 0) + 1
        confs.setdefault(d.cls_name, []).append(float(d.confidence))
    return {
        "total": len(detections),
        "by_class": {c: {"count": counts[c], "mean_conf": round(sum(confs[c]) / len(confs[c]), 3)}
                     for c in counts},
    }


def _kb_for_classes(kb: dict, classes: Sequence[str], lang: str) -> dict:
    out = {}
    for c in classes:
        entry = kb.get("classes", {}).get(c)
        if entry:
            out[c] = entry.get(lang, entry.get("en", {}))
    return out


def build_prompt(detections: Sequence[Detection], kb: dict, *, lang: str,
                 xai_summary: str | None = None, image_meta: dict | None = None) -> tuple[str, str]:
    """Return (system, user_prompt) grounding the LLM in the KB + detections."""
    classes = sorted({d.cls_name for d in detections})
    payload = {
        "language": LANG_NAME.get(lang, "English"),
        "image_metadata": image_meta or {},
        "detection_summary": _detection_summary(detections),
        "detections": [{"class": d.cls_name, "confidence": round(float(d.confidence), 3),
                        "bbox_xyxy": [round(float(x), 1) for x in d.bbox]} for d in detections],
        "xai_attention_summary": xai_summary or "Not provided.",
        "knowledge_base": _kb_for_classes(kb, classes, lang),
        "required_sections": SECTIONS.get(lang, SECTIONS["en"]),
    }
    system = (
        "You are a metallurgical quality-assurance assistant for steel surface "
        "inspection. Write a precise, professional inspection report. Use ONLY the "
        "provided knowledge_base and detections - never invent defects, causes, or "
        "numbers that are not given. If no defects were detected, say so plainly. "
        f"Write the ENTIRE report in {LANG_NAME.get(lang, 'English')}."
    )
    user = (
        "Produce the report with exactly these sections (use them as headings, in "
        f"order): {payload['required_sections']}.\n\n"
        "Structured input (JSON):\n```json\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    return system, user


# --------------------------------------------------------------------------- #
# Deterministic fallback (KB-only, no LLM) - used when the endpoint is down
# --------------------------------------------------------------------------- #
def _template_report(detections: Sequence[Detection], kb: dict, lang: str,
                     xai_summary: str | None, image_meta: dict | None) -> str:
    secs = SECTIONS.get(lang, SECTIONS["en"])
    summ = _detection_summary(detections)
    classes = sorted({d.cls_name for d in detections})
    kbl = _kb_for_classes(kb, classes, lang)
    L = lang == "ar"

    lines: list[str] = []
    # 1. Executive summary
    lines.append(f"## {secs[0]}")
    if not detections:
        lines.append("لم يتم اكتشاف أي عيوب فوق عتبة الثقة." if L
                     else "No defects were detected above the confidence threshold.")
    else:
        head = (f"تم اكتشاف {summ['total']} عيبًا عبر {len(classes)} فئة: "
                if L else f"Detected {summ['total']} defect(s) across {len(classes)} class(es): ")
        parts = [f"{kbl.get(c, {}).get('name', c)} ×{summ['by_class'][c]['count']} "
                 f"({'متوسط ثقة' if L else 'mean conf'} {summ['by_class'][c]['mean_conf']:.0%})"
                 for c in classes]
        lines.append(head + "؛ ".join(parts) + ".")
    # 2. Per-defect explanation
    lines.append(f"\n## {secs[1]}")
    for c in classes:
        e = kbl.get(c, {})
        lines.append(f"### {e.get('name', c)}")
        lines.append(("**التعريف:** " if L else "**Definition:** ") + e.get("definition", ""))
        lines.append(("**السبب الجذري المحتمل:** " if L else "**Likely root cause:** ") + e.get("root_cause", ""))
        lines.append(("**العلامة البصرية:** " if L else "**Visual signature:** ") + e.get("visual_signature", ""))
        lines.append(("**الخطورة:** " if L else "**Severity:** ")
                     + f"{e.get('severity', '')} - {e.get('severity_note', '')}")
    # 3. XAI
    lines.append(f"\n## {secs[2]}")
    lines.append(xai_summary or ("لم يتم توفير ملخص الانتباه." if L else "No XAI attention summary was provided."))
    # 4. Recommended action
    lines.append(f"\n## {secs[3]}")
    for c in classes:
        e = kbl.get(c, {})
        lines.append(f"- **{e.get('name', c)}:** {e.get('recommended_action', '')}")
    if not classes:
        lines.append("لا إجراء مطلوب." if L else "No action required.")
    # 5. Caveats
    lines.append(f"\n## {secs[4]}")
    lines.append(
        ("هذا تقرير احتياطي مُولَّد من قاعدة المعرفة فقط (نموذج اللغة غير متاح). "
         "تستند الاكتشافات إلى ثقة النموذج وقد تتضمن أخطاء؛ يلزم تأكيد بشري.") if L else
        ("This is a fallback report generated from the knowledge base only (the LLM "
         "endpoint was unavailable). Detections reflect model confidence and may contain "
         "errors; human confirmation is required.")
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def generate_report(detections: Sequence[Detection], *, lang: str = "en",
                    xai_summary: str | None = None, image_meta: dict | None = None,
                    provider: OpenAICompatProvider | None = None,
                    kb: dict | None = None) -> dict:
    """Generate one report in `lang`. Falls back to a KB-only template if the LLM
    endpoint is unreachable. Returns a dict with text + provenance."""
    kb = kb if kb is not None else load_kb()
    system, user = build_prompt(detections, kb, lang=lang, xai_summary=xai_summary, image_meta=image_meta)

    text, used_llm, error = None, False, None
    prov = provider if provider is not None else default_provider()
    try:
        text = prov.generate(user, system=system)
        used_llm = bool(text)
    except Exception as exc:  # connection refused, timeout, missing openai, etc.
        error = f"{type(exc).__name__}: {exc}"
    if not text:
        text = _template_report(detections, kb, lang, xai_summary, image_meta)

    return {
        "lang": lang,
        "used_llm": used_llm,
        "error": error,
        "text": text,
        "summary": _detection_summary(detections),
        "detections": [{"class": d.cls_name, "confidence": float(d.confidence),
                        "bbox": list(d.bbox)} for d in detections],
    }


def generate_bilingual(detections: Sequence[Detection], **kwargs) -> dict:
    """Generate EN + AR reports in one call. kwargs pass through to generate_report."""
    return {lang: generate_report(detections, lang=lang, **kwargs) for lang in ("en", "ar")}


# --------------------------------------------------------------------------- #
# Export: HTML + PDF
# --------------------------------------------------------------------------- #
def _md_inline_to_html(text: str) -> str:
    """Tiny markdown-ish -> HTML (headings, bold, bullets). Avoids a markdown dep."""
    out, in_ul = [], False
    for raw in text.splitlines():
        line = raw.rstrip()
        esc = html.escape(line)
        # bold
        while "**" in esc:
            esc = esc.replace("**", "<strong>", 1).replace("**", "</strong>", 1)
        if line.startswith("### "):
            out.append(f"<h3>{esc[4:]}</h3>"); _close_ul(out, in_ul); in_ul = False
        elif line.startswith("## "):
            out.append(f"<h2>{esc[3:]}</h2>"); _close_ul(out, in_ul); in_ul = False
        elif line.startswith("- "):
            if not in_ul:
                out.append("<ul>"); in_ul = True
            out.append(f"<li>{esc[2:]}</li>")
        elif not line:
            _close_ul(out, in_ul); in_ul = False
        else:
            _close_ul(out, in_ul); in_ul = False
            out.append(f"<p>{esc}</p>")
    _close_ul(out, in_ul)
    return "\n".join(out)


def _close_ul(out: list, in_ul: bool):
    if in_ul:
        out.append("</ul>")


def to_html(report: dict, *, title: str | None = None) -> str:
    """Render one report dict to a standalone HTML string (Arabic uses dir=rtl)."""
    lang = report.get("lang", "en")
    rtl = lang == "ar"
    title = title or ("تقرير فحص عيوب سطح الفولاذ" if rtl else "Steel Surface Defect Inspection Report")
    badge = ("" if report.get("used_llm") else
             ('<div class="badge">' + ("تقرير احتياطي (نموذج اللغة غير متاح)" if rtl
              else "Fallback report (LLM unavailable)") + "</div>"))
    body = _md_inline_to_html(report.get("text", ""))
    align = "right" if rtl else "left"
    return f"""<!DOCTYPE html>
<html lang="{lang}" dir="{'rtl' if rtl else 'ltr'}">
<head><meta charset="utf-8"><title>{html.escape(title)}</title>
<style>
 body{{font-family:'Segoe UI',Tahoma,Arial,sans-serif;max-width:820px;margin:24px auto;
       padding:0 18px;line-height:1.6;color:#1a1a1a;text-align:{align};}}
 h1{{border-bottom:3px solid #c0392b;padding-bottom:6px;}}
 h2{{color:#c0392b;margin-top:1.4em;}} h3{{color:#34495e;}}
 .badge{{display:inline-block;background:#f39c12;color:#fff;padding:3px 10px;border-radius:4px;font-size:.85em;}}
 ul{{margin:.3em 0;}}
</style></head>
<body><h1>{html.escape(title)}</h1>{badge}
{body}
</body></html>"""


def save_html(report: dict, path: str | Path, *, title: str | None = None) -> Path:
    path = Path(path)
    path.write_text(to_html(report, title=title), encoding="utf-8")
    return path


def _shape_ar(s: str) -> str:
    """Reshape + reorder Arabic for engines (reportlab) that don't do RTL shaping."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        return get_display(arabic_reshaper.reshape(s))
    except Exception:
        return s


def _render_pdf(report: dict, target, *, title: str | None = None,
                ar_font_path: str = "C:/Windows/Fonts/arial.ttf"):
    """Render one report to `target` (a path string or a writable buffer) via
    reportlab. Arabic is shaped + right-aligned. Raises if reportlab is unavailable."""
    from reportlab.lib.enums import TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

    lang = report.get("lang", "en")
    rtl = lang == "ar"
    font = "Helvetica"
    if rtl and Path(ar_font_path).exists():
        try:
            pdfmetrics.registerFont(TTFont("ARFont", ar_font_path))
            font = "ARFont"
        except Exception:
            pass
    align = TA_RIGHT if rtl else TA_LEFT

    def P(text: str, size: float = 10.5, bold: bool = False, space: int = 4) -> Paragraph:
        shaped = _shape_ar(text) if rtl else text
        style = ParagraphStyle("s", fontName=font, fontSize=size, leading=size * 1.5,
                               alignment=align, spaceAfter=space,
                               textColor="#c0392b" if bold else "#1a1a1a")
        safe = shaped.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return Paragraph(f"<b>{safe}</b>" if bold else safe, style)

    ttl = title or ("تقرير فحص عيوب سطح الفولاذ" if rtl else "Steel Surface Defect Inspection Report")
    story = [P(ttl, size=16, bold=True, space=10)]
    if not report.get("used_llm"):
        story.append(P("(تقرير احتياطي - نموذج اللغة غير متاح)" if rtl
                       else "(Fallback report - LLM unavailable)", size=9))
    story.append(Spacer(1, 0.3 * cm))
    for raw in report.get("text", "").splitlines():
        line = raw.strip()
        if not line:
            story.append(Spacer(1, 0.15 * cm)); continue
        plain = line.lstrip("#").replace("**", "").strip()
        if line.startswith("##"):
            story.append(P(plain, size=13 if line.startswith("## ") else 11.5, bold=True, space=6))
        elif line.startswith("- "):
            story.append(P("•  " + plain[2:], size=10.5))
        else:
            story.append(P(plain, size=10.5))

    dst = target if hasattr(target, "write") else str(target)
    SimpleDocTemplate(dst, pagesize=A4, title=ttl,
                      leftMargin=2 * cm, rightMargin=2 * cm,
                      topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(story)
    return target


def save_pdf(report: dict, path: str | Path, *, title: str | None = None,
             ar_font_path: str = "C:/Windows/Fonts/arial.ttf") -> Path:
    """Render one report to a PDF file (callers may fall back to save_html on error)."""
    path = Path(path)
    _render_pdf(report, str(path), title=title, ar_font_path=ar_font_path)
    return path


def to_pdf_bytes(report: dict, *, title: str | None = None,
                 ar_font_path: str = "C:/Windows/Fonts/arial.ttf") -> bytes:
    """Render one report to PDF and return the raw bytes (for Streamlit downloads)."""
    import io

    buf = io.BytesIO()
    _render_pdf(report, buf, title=title, ar_font_path=ar_font_path)
    return buf.getvalue()


def save_report(report: dict, stem: str | Path) -> dict:
    """Save both HTML and (best-effort) PDF. Returns the written paths."""
    stem = Path(stem)
    out = {"html": str(save_html(report, stem.with_suffix(".html")))}
    try:
        out["pdf"] = str(save_pdf(report, stem.with_suffix(".pdf")))
    except Exception as exc:
        out["pdf"] = None
        out["pdf_error"] = f"{type(exc).__name__}: {exc}"
    return out
