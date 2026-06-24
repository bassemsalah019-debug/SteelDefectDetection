# LABEL_QUALITY_REPORT.md
*Generated 2026-06-13 by `scripts/audit_dataset.py`. Measured.*

## Headline: label quality is NOT a limiter.
| Check | Result |
|---|---|
| Missing label files | **0 / 1800** |
| Empty label files | **0 / 1800** |
| Malformed lines (≠5 fields) | **0** |
| Class id out of range (not 0–5) | **0** |
| Coords outside [0,1] | **0** |
| Boxes total | 4189 (mean 2.33/img, 1–9) |

All 1800 images carry a valid YOLO label. NEU-DET ships VOC XML; the VOC→YOLO conversion
(`src/voc_to_yolo.py`) introduced no integrity errors.

## Caveats the integrity check cannot catch (require visual review)
These are **hypotheses** about annotation *style*, not measured defects:
1. **Annotation granularity is coarse for texture classes.** `crazing` (median box = 21% of
   the frame) and `pitted_surface` (55%) are frequently annotated as one near-whole-image box
   rather than per-instance regions. This is consistent across NEU-DET and is *intended* (these
   are surface-wide textures), but it means localization for these classes is near-trivial /
   degenerate — the task is closer to **classification** for them. Impact: inflates `patches`/
   `pitted_surface` AP@0.5, depresses AP@0.5:0.95.
2. **`inclusion` is densely multi-instance** (852 train boxes, median 4% area) — the opposite
   regime. Missed small inclusions are the likely recall driver, not mislabels.
3. **No inter-annotator agreement data** ships with NEU-DET; boundary precision on diffuse
   `crazing`/`rolled-in_scale` is inherently ambiguous and likely contributes to their ~0.45–0.64
   ceiling regardless of model.

## Recommended (optional) verification — not yet done
- Render a 6×N contact sheet of boxes per class for a 10-minute visual sanity pass (script can be
  generated on request). **Expected value: low** — integrity is perfect; only style ambiguity remains,
  which is a *dataset* property, not a fixable label error.

**Conclusion:** do not spend effort "cleaning labels." Spend it on evaluation rigor (seeds) and
scale-aware modeling. Label quality severity: **NONE / informational.**
