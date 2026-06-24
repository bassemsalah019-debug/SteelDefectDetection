"""LLM defect-report package (grounded, bilingual EN/AR, HTML+PDF export)."""
from src.report.report_generator import (
    Detection,
    OpenAICompatProvider,
    build_prompt,
    default_provider,
    generate_bilingual,
    generate_report,
    load_kb,
    save_html,
    save_pdf,
    save_report,
    to_html,
    to_pdf_bytes,
)

__all__ = [
    "Detection",
    "OpenAICompatProvider",
    "build_prompt",
    "default_provider",
    "generate_bilingual",
    "generate_report",
    "load_kb",
    "save_html",
    "save_pdf",
    "save_report",
    "to_html",
    "to_pdf_bytes",
]
