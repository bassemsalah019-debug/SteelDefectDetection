"""
Report-generator tests - exercise the grounded, bilingual report WITHOUT a live
LLM (a failing provider forces the deterministic KB-only fallback), plus HTML/PDF
export. No network, no GPU, so this runs in CI.

    python -m pytest tests/test_report_fallback.py -q
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.report import (  # noqa: E402
    Detection,
    build_prompt,
    generate_bilingual,
    generate_report,
    load_kb,
    save_report,
    to_html,
)


class _FailingProvider:
    """Stand-in for an unreachable endpoint - forces the fallback path."""

    def generate(self, prompt, system=None):
        raise ConnectionError("endpoint down (test)")


DETS = [
    Detection("scratches", 0.91, [10, 12, 80, 18]),
    Detection("inclusion", 0.77, [40, 50, 60, 70]),
    Detection("scratches", 0.66, [5, 90, 70, 96]),
]


def test_kb_has_six_classes_with_bilingual_fields():
    kb = load_kb()
    classes = kb["classes"]
    assert set(classes) == {"crazing", "inclusion", "patches",
                            "pitted_surface", "rolled-in_scale", "scratches"}
    for c, entry in classes.items():
        for lang in ("en", "ar"):
            for field in ("definition", "root_cause", "visual_signature",
                          "severity", "recommended_action"):
                assert entry[lang][field].strip(), f"{c}.{lang}.{field} empty"


def test_fallback_used_when_provider_fails():
    rep = generate_report(DETS, lang="en", provider=_FailingProvider())
    assert rep["used_llm"] is False
    assert rep["error"] is not None
    assert rep["text"].strip()
    # grounded in the KB, mentions detected classes only
    assert "Scratches" in rep["text"] and "Inclusion" in rep["text"]
    assert "Crazing" not in rep["text"]  # never detected -> not explained
    assert rep["summary"]["total"] == 3


def test_fallback_arabic_is_arabic():
    rep = generate_report(DETS, lang="ar", provider=_FailingProvider())
    assert rep["used_llm"] is False
    assert any("؀" <= ch <= "ۿ" for ch in rep["text"]), "no Arabic chars"


def test_no_detections_reports_clean():
    rep = generate_report([], lang="en", provider=_FailingProvider())
    assert "No defects" in rep["text"]
    assert rep["summary"]["total"] == 0


def test_build_prompt_grounds_only_detected_classes():
    system, user = build_prompt(DETS, load_kb(), lang="en")
    assert "metallurgical" in system.lower()
    assert "scratches" in user and "inclusion" in user
    assert "crazing" not in user  # KB slice is limited to detected classes


def test_generate_bilingual_keys():
    both = generate_bilingual(DETS, provider=_FailingProvider())
    assert set(both) == {"en", "ar"}
    assert both["en"]["lang"] == "en" and both["ar"]["lang"] == "ar"


def test_to_html_rtl_for_arabic():
    rep = generate_report(DETS, lang="ar", provider=_FailingProvider())
    doc = to_html(rep)
    assert 'dir="rtl"' in doc and "<html" in doc


def test_save_report_writes_html_and_best_effort_pdf(tmp_path):
    rep = generate_report(DETS, lang="en", provider=_FailingProvider())
    out = save_report(rep, tmp_path / "report_en")
    assert Path(out["html"]).exists() and Path(out["html"]).stat().st_size > 0
    # PDF is best-effort: either a written file or a captured error (never a crash)
    assert ("pdf" in out) and (out["pdf"] is None or Path(out["pdf"]).exists())


def test_to_pdf_bytes_is_a_pdf():
    from src.report import to_pdf_bytes

    rep = generate_report(DETS, lang="en", provider=_FailingProvider())
    data = to_pdf_bytes(rep)
    assert isinstance(data, bytes) and data[:5] == b"%PDF-"
