# Production model

`best.pt` — the **YOLOv8n baseline @640**, the production/deployed model for this project.

| | |
|---|---|
| Architecture | YOLOv8n (stock) |
| Task | Steel surface defect detection — 6 classes (NEU-DET) |
| TEST mAP@0.5 | **0.7525** (5-seed mean **0.7475 ± 0.016**) |
| Size · speed | 6.0 MB · ~157 FPS (RTX 2000 Ada) |
| Chosen over | Ghost+MPCA+SIoU, LZY (CBAM+WIoU), P2-head, YOLOv8s — none beat it under a fair 5-seed gate |

This is the exact checkpoint served by both demos (Streamlit Studio and the SteelVision app).
Full details and the honest evaluation are in **[../docs/model_card.md](../docs/model_card.md)** and
**[../docs/audit/](../docs/audit/)**. Classes: crazing, inclusion, patches, pitted_surface,
rolled-in_scale, scratches.
