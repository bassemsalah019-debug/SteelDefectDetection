"""
Custom modules that reproduce the improved YOLOv8n from
"A lightweight algorithm for steel surface defect detection using improved
YOLOv8", Scientific Reports (2025), s41598-025-93469-5.

The paper makes three changes to YOLOv8n:
  1. Ghost backbone  -> GhostConv + C3Ghost  (both already in ultralytics).
  2. MPCA attention  -> end of backbone       (this package, see mpca.py).
  3. SIoU loss        -> replaces CIoU         (this package, see siou.py).

`register()` makes (2) and (3) usable WITHOUT editing the installed ultralytics
package:
  - MPCA is injected into `ultralytics.nn.tasks`' namespace so a model YAML can
    reference it by name (parse_model resolves custom layers via `globals()[m]`).
  - BboxLoss.forward is monkey-patched to compute SIoU instead of CIoU.

A SECOND improved model (the GitHub project LZY-233/yolov8_Imporved-Defect_detection:
GhostConv backbone + ResBlock_CBAM head + WIoU loss) is wired by `register_lzy()`:
  - ResBlock_CBAM is injected into `ultralytics.nn.tasks` so a YAML can name it.
  - BboxLoss.forward is monkey-patched to compute WIoU instead of CIoU.

Usage:
    import sys; sys.path.insert(0, str(ROOT))      # repo root on path
    from src.modules import register               # paper model (notebook 05)
    register()                                      # MPCA + SIoU active
    # --- or, for the LZY-233 model (notebook 06) ---
    from src.modules import register_lzy
    register_lzy()                                  # ResBlock_CBAM + WIoU active
"""

from .mpca import MPCA
from .siou import bbox_siou, patch_bbox_loss_with_siou
from .rescbam import ResBlock_CBAM, CBAM, ChannelAttention, SpatialAttention
from .wiou import bbox_wiou, patch_bbox_loss_with_wiou

__all__ = ["MPCA", "bbox_siou", "register", "ResBlock_CBAM", "bbox_wiou", "register_lzy"]


def register(mpca: bool = True, siou: bool = True, verbose: bool = True) -> None:
    """Register the paper's custom modules with Ultralytics (idempotent).

    Args:
        mpca: make the ``MPCA`` layer name resolvable inside model YAMLs.
        siou: replace the CIoU regression loss with SIoU.
        verbose: print a one-line confirmation of what was activated.
    """
    activated = []

    if mpca:
        # parse_model() looks up custom layer names in its own module globals,
        # so binding MPCA there lets `[-1, 1, MPCA, []]` build correctly.
        import ultralytics.nn.tasks as tasks

        tasks.MPCA = MPCA
        # torch>=2.6 defaults torch.load(weights_only=True); allow-list the custom
        # classes so reloading a saved improved best.pt unpickles cleanly.
        try:
            import torch
            from .mpca import _CoordAttPath

            torch.serialization.add_safe_globals([MPCA, _CoordAttPath])
        except Exception:
            pass
        activated.append("MPCA (end-of-backbone attention)")

    if siou:
        patch_bbox_loss_with_siou()
        activated.append("SIoU regression loss")

    if verbose:
        print("[src.modules.register] activated:", ", ".join(activated) if activated else "nothing")


def register_lzy(rescbam: bool = True, wiou: bool = True, verbose: bool = True) -> None:
    """Register the LZY-233 model's custom modules with Ultralytics (idempotent).

    Args:
        rescbam: make the ``ResBlock_CBAM`` layer name resolvable inside model YAMLs.
        wiou: replace the CIoU regression loss with WIoU.
        verbose: print a one-line confirmation of what was activated.
    """
    activated = []

    if rescbam:
        import ultralytics.nn.tasks as tasks

        tasks.ResBlock_CBAM = ResBlock_CBAM
        # allow-list the custom classes so reloading a saved best.pt unpickles cleanly
        try:
            import torch

            torch.serialization.add_safe_globals(
                [ResBlock_CBAM, CBAM, ChannelAttention, SpatialAttention]
            )
        except Exception:
            pass
        activated.append("ResBlock_CBAM (head attention)")

    if wiou:
        patch_bbox_loss_with_wiou()
        activated.append("WIoU regression loss")

    if verbose:
        print("[src.modules.register_lzy] activated:", ", ".join(activated) if activated else "nothing")
