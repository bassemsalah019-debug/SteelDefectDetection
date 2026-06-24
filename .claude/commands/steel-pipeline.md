---
description: Run the steel defect detection pipeline phase by phase with approval gates.
---
Drive the steel surface defect detection project through Phases 0–8 of the master prompt (`CLAUDE_CODE_MASTER_PROMPT.md`), adapted to this repo's real layout.

Topology note: this repo IS the project, built directly at the root (`src/`, `configs/`, `notebooks/`, `deployment/`, `docs/`, `tests/`, `results/`, `runs/`). There is NO separate read-only "original" and NO `updated_project/` folder — ignore those paths from the master prompt and work in the real folders. Substantial work already exists (Eigen-CAM XAI, HF Space, notebooks through `updated_07`, real runs in `results/` with best `improved_opt` ≈ 0.7678), so Phase 0 ASSESSES current state against the definition of done rather than assuming a blank repo.

Rules:
- Enforce every HARD CONSTRAINT and every approval gate. Never cross a gate without the user's explicit "go".
- SINGLE GPU (RTX 2000 Ada, 16 GB): all GPU work — training, eval, INT8 calibration, TensorRT builds — runs sequentially, one job at a time. The user runs GPU/training jobs themselves in the notebooks; agents prepare configs + cells but never launch GPU work.
- Fan out only genuinely independent NON-GPU work to subagents (config generation, results analysis, XAI/report/app code, docs). Keep dependent steps sequential. Prefer one well-scoped subagent over a chatty team.
- No fabricated numbers, ever. Every metric must trace to a real saved results file.
- Before each gate, post a short plan (scope, time, cost, what changes). Where relevant, have `reviewer` check correctness, preprocessing parity, and metric honesty.
- After each phase, update `docs/PROGRESS.md` and keep `experiments/LEADERBOARD.md` sorted by mAP@0.5 descending.

Phase roster: Phase 0 recon → `repo-auditor` (read-only); Phase 1 data/parity → `data-engineer`; Phase 2 baseline + harness → `experiment-runner`; Phase 3 optimization campaign → `experiment-runner`; Phase 4 explainability → `xai-engineer`; Phase 5 LLM report → `report-engineer`; Phase 6 app + grayscale fix → `app-engineer`; Phase 7 export/benchmark → `deploy-engineer`; Phase 8 publish/docs → `deploy-engineer` + `reviewer`.

Start with Phase 0 (`repo-auditor`, read-only) and STOP at Gate 0 with the audit + plan.
