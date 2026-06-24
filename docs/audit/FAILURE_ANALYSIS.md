# FAILURE_ANALYSIS.md
*Generated 2026-06-13. Per-class numbers are measured (TEST @640, `results/*/results.json`).
Per-image FP/FN clustering is scaffolded but not yet run (needs a prediction dump — see §4).*

## 1. Per-class TEST mAP@0.5 (@640, measured today)
| class | baseline | paper | LZY | median box area | verdict |
|---|---|---|---|---|---|
| patches | 0.925 | 0.934 | 0.935 | 0.093 | solved |
| pitted_surface | 0.862 | 0.844 | 0.869 | 0.555 | strong (large boxes, easy IoU) |
| inclusion | 0.858 | 0.842 | 0.804 | **0.041** | strong but small-object recall-limited |
| scratches | 0.794 | 0.773 | 0.818 | 0.075 | resolution-sensitive |
| **rolled-in_scale** | 0.636 | 0.574 | 0.521 | 0.123 | **weak** |
| **crazing** | 0.440 | 0.416 | 0.444 | 0.211 | **floor (all models)** |

## 2. Root-cause clusters (evidence-linked)
**Cluster A — `crazing` floor (~0.42–0.44, architecture-invariant).** Lowest AP for every model
and recipe. Crazing = diffuse hairline-crack network, low contrast, annotated as large regions.
The model card notes Eigen-CAM finds "no localized signal." Root cause = **low signal-to-noise +
degenerate localization target**, not capacity. *Architecture cannot fix this; contrast/data can.*

**Cluster B — `rolled-in_scale` second-weakest, and the class the mods HURT most**
(baseline 0.636 → paper 0.574 → LZY 0.521). Low-contrast embedded scale; the Ghost backbone's
reduced channel capacity and CBAM/WIoU appear to drop exactly this low-contrast class. Root cause =
**low contrast** + **architecture compression removing fine texture features**.

**Cluster C — small-object recall on `inclusion`** (median 4% area, 852 instances). High AP but the
class most helped by resolution (640→800). Root cause = **small objects under-resolved**; addressable
by P2 head / higher input res.

**Cluster D — `scratches` resolution sensitivity** (thin, elongated, 7.5% area). Documented to jump
at higher resolution. Root cause = **thin features lost at low res / by aggressive downsampling**.

## 3. Why the "improved" architectures lose
The baseline's +2.1 pp @640 is almost entirely **rolled-in_scale (+6.1 pp vs paper)** and
**inclusion (+1.6 pp vs paper, +5.4 vs LZY)** — both **low-contrast / small** classes. The
lightweight backbones trade away exactly the fine-feature capacity these classes need. patches/
pitted_surface (large, easy) are unaffected. **This is a coherent, physical explanation**, not noise.

## 4. Deeper analysis prepared (not yet run)
A prediction-level error analysis (confusion matrix from predictions, FP/FN per image, hardest
images) needs a prediction dump on the test set. Suggested next step (one GPU eval, ~1 min):
`yolo val ... save_json=True` per model → cluster by IoU/confidence. Hypotheses to test:
- **`inclusion`↔`pitted_surface` confusion** (the leakage check showed these two share global
  texture; correlation ≥0.97 cross-class pairs were almost all between these classes).
- **`crazing` false-negatives** dominate vs false-positives (low recall, not over-prediction).

I can generate `scripts/error_analysis.py` to produce these on request.

## Severity / leverage
| Failure | Severity | Best lever | Architecture-fixable? |
|---|---|---|---|
| crazing floor | HIGH | CLAHE/contrast, more data, relabel granularity | **No** |
| rolled-in_scale | HIGH | keep full-capacity backbone, contrast aug | partly (don't compress) |
| inclusion small-obj | MED | P2 head, higher res | yes |
| scratches | MED | higher res, mosaic tuning | yes |
