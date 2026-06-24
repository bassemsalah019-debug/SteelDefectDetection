# PROGRESS

Phase-by-phase log per the master prompt (`CLAUDE_CODE_MASTER_PROMPT.md`), adapted
to this repo's real layout (work happens at root; there is no `updated_project/`).

---

## Tooling bootstrap (2026-06-10)
- Created `.claude/agents/` (repo-auditor, data-engineer, experiment-runner,
  xai-engineer, report-engineer, app-engineer, deploy-engineer, reviewer) and
  `.claude/commands/steel-pipeline.md`, adapted to the real root layout.

## Phase 0 - Recon (done)
- Read app, XAI, modules, configs, tests, and every `results/<run>/metrics_summary.txt`.
- Current best: `improved_opt`, **TEST mAP@0.5 = 0.7678** (paper improved 0.786).
- Located the RGB->grayscale bug: `src/app.py` and `src/explain.py` fed colour RGB;
  NEU-DET training saw grayscale-replicated 3-channel input.
- **Honesty finding:** `docs/PROJECT_AUDIT.md` section-2 @640 numbers don't match the
  saved `metrics_summary.txt` files (see `experiments/LEADERBOARD.md` > Discrepancy).

## Phase 1 - Preprocessing parity + bug fix (done)
Files created:
- `src/preprocessing.py` - canonical `to_model_input(img) -> (H,W,3) uint8` grayscale-
  replicated, using PIL's exact L24 fixed-point luma so PIL/numpy paths are byte-identical.
  Owns only the colour->gray collapse; Ultralytics still does letterbox + 0-1 norm.
- `src/infer.py` - the one inference wrapper (`preprocess`, `predict`) shared by app + eval.
- `tests/test_preprocessing_parity.py` - 8 tests: app path == inference path (byte-identical),
  output is grayscale-replicated, colour input collapses, BGR flag, PIL==numpy, matches
  PIL `.convert("L")`.

Files edited (bug fixed at source):
- `src/app.py` - detection now routes through `run_detection` (canonical grayscale);
  added a visible "✓ Input collapsed to grayscale (NEU-DET parity)" indicator.
- `src/explain.py` - `_to_input` now grayscale-replicates via `to_model_input` so the
  Eigen-CAM input matches the detector.

Commands run + result:
```
.venv/Scripts/python -m pytest tests/test_preprocessing_parity.py -q   -> 8 passed
.venv/Scripts/python -m pytest tests/ -q                               -> 13 passed (no regressions)
```

Deliverable created: `experiments/LEADERBOARD.md` (all rows traced to saved metrics files).

## Phase 5 - LLM defect report (code done; live LLM gated)
Files created:
- `src/report/knowledge_base.json` + `.md` - grounded KB for all 6 classes, bilingual
  EN/AR (definition, root_cause, visual_signature, severity, recommended_action).
- `src/report/report_generator.py`:
  - Swappable provider `OpenAICompatProvider.generate(prompt, system)`; base_url/model/
    api_key from env (`STEEL_LLM_BASE_URL` default Ollama `:11434/v1`, `STEEL_LLM_MODEL`
    default `mimo-7b-rl`, `STEEL_LLM_API_KEY`). Nothing hardcoded.
  - Grounded prompt (detections + KB slice for detected classes only + XAI summary +
    image meta); LLM told to explain ONLY detected defects.
  - **Graceful degrade:** if the endpoint is unreachable -> deterministic KB-only
    template report, clearly marked. App never breaks.
  - Bilingual `generate_bilingual` (EN+AR); export `to_html`/`save_html` and
    `save_pdf` (reportlab; Arabic shaped RTL via arabic_reshaper + python-bidi);
    `save_report` writes both (PDF best-effort).
- `tests/test_report_fallback.py` - 8 tests (offline fallback, bilingual, grounding,
  HTML/PDF), all green.
Verified: EN + AR HTML and PDF render (AR PDF embeds the Arabic TTF). To use the real
LLM: serve MiMo-7B-RL via Ollama and set the env vars (separate download/serve gate).

## Phase 2 (partial) - eval/measurement harness (code done; running it is a GPU gate)
- `src/eval.py` - `evaluate(weights, split='test', imgsz, device)` runs `model.val`,
  collects mAP@0.5, mAP@0.5:0.95, P, R, per-class AP@0.5, params, GFLOPs (ultralytics
  get_flops), FPS (timed on the stated device), model size -> writes `results.json`.
  CLI: `python -m src.eval --weights ... --device 0`. Imports cleanly without a GPU.
  Run it per checkpoint to fill the leaderboard's GFLOPs/FPS columns (your GPU).

## Phase 4 (partial) + Phase 6 - XAI attention summary + app report integration (done)
- `src/explain.py` - added `attention_summary(heat, detections, img_size)`: a one-line
  structured description of WHERE Eigen-CAM looks (region + coverage % + peak + how many
  detections fall in the hot zone). Numpy-only; feeds the LLM report.
- `src/report/report_generator.py` - refactored PDF into `_render_pdf` + added
  `to_pdf_bytes` (in-memory PDF for Streamlit download).
- `src/app.py` - added the end-to-end report section: EN/AR language toggle, builds
  `Detection`s from the boxes, passes the XAI attention summary + image metadata,
  generates the grounded report (LLM if configured, else KB fallback), renders it, and
  offers **HTML + PDF download** buttons. Full flow: upload -> grayscale parity -> detect
  -> Eigen-CAM -> bilingual report -> PDF.

Tests added: `attention_summary` (2) + `to_pdf_bytes` is a real PDF (1).
Full suite: `.venv/Scripts/python -m pytest tests/ -q` -> **24 passed**. `app.py` compiles.

## Open gates (NOT crossed - awaiting "go")
- **GPU training** (Phase 2/3): attempt to beat 0.786, mean±std over >=3 seeds. You run
  these in the notebooks; experiment-runner only prepares configs/cells.
- **Eval harness** (Phase 2): measure params/GFLOPs/FPS on the RTX 2000 Ada for the leaderboard.
- **MiMo / Ollama download + serve** (Phase 5): local LLM report generator.
- **Grad-CAM/D-RISE deps** (Phase 4): `pip install grad-cam` (env change -> gate).
- **TensorRT toolchain + engine build** (Phase 7): download + GPU build.
- **Publish** (Phase 8): any external push (HF Spaces).
