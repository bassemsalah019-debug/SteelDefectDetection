"""
SIoU bounding-box regression loss.

Reproduces the loss change from:
  "A lightweight algorithm for steel surface defect detection using improved
   YOLOv8", Scientific Reports (2025), s41598-025-93469-5.

Paper (Methods): the baseline CIoU regression loss is replaced with SIoU
(Gevorgyan, 2022). SIoU adds an ANGLE cost so the predicted box is first pulled
onto the nearest x/y axis relative to the target, which speeds convergence:

    Loss_SIoU = 1 - IoU + (Delta + Omega) / 2

    Angle cost     Lambda = cos(2 * (arcsin(x) - pi/4)),  x = c_h / sigma
    Distance cost  Delta  = sum_{t in x,y} (1 - exp(-(2 - Lambda) * rho_t))
    Shape  cost    Omega  = sum_{t in w,h} (1 - exp(-w_t)) ** theta

Faithfulness notes:
  - [PAPER]   SIoU replaces CIoU; formula above (angle + distance + shape).
  - [DEFAULT] shape-cost exponent theta = 4 (Gevorgyan's recommended value;
              the paper does not print an explicit theta).

Ultralytics 8.4.51 has no SIoU option, so this module provides `bbox_siou` and
monkey-patches `ultralytics.utils.loss.BboxLoss.forward` to use it in place of
the CIoU term. The DFL term is left untouched. Activated via
`src.modules.register(siou=True)`; no edits to the installed package.
"""

import math

import torch
import torch.nn.functional as F

# SIoU shape-cost exponent (Gevorgyan 2022 recommends theta in [2, 6], default 4).
SIOU_THETA = 4.0


def bbox_siou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """Compute SIoU between two sets of boxes.

    Returns ``IoU - (Delta + Omega) / 2`` so the caller's ``1 - value`` matches
    the published ``Loss_SIoU = 1 - IoU + (Delta + Omega) / 2``. Shape mirrors
    Ultralytics' ``bbox_iou`` (trailing singleton dim preserved).
    """
    if xywh:  # (cx, cy, w, h) -> (x1, y1, x2, y2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # (x1, y1, x2, y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection / union -> IoU
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # Smallest enclosing box
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # enclosing width
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # enclosing height

    # Centre offsets between the two boxes
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5      # x centre distance
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5      # y centre distance
    sigma = (s_cw.pow(2) + s_ch.pow(2)).sqrt() + eps  # centre distance

    # --- Angle cost Lambda --------------------------------------------------
    sin_alpha_1 = (s_cw.abs() / sigma).clamp(0, 1)    # angle to x-axis
    sin_alpha_2 = (s_ch.abs() / sigma).clamp(0, 1)    # angle to y-axis
    threshold = math.sin(math.pi / 4)                 # snap to the nearer axis
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    angle_cost = torch.cos(2 * (torch.arcsin(sin_alpha) - math.pi / 4))  # Lambda

    # --- Distance cost Delta -------------------------------------------------
    rho_x = (s_cw / cw.clamp(min=eps)).pow(2)
    rho_y = (s_ch / ch.clamp(min=eps)).pow(2)
    gamma = 2 - angle_cost                            # (2 - Lambda)
    distance_cost = (1 - torch.exp(-gamma * rho_x)) + (1 - torch.exp(-gamma * rho_y))

    # --- Shape cost Omega ----------------------------------------------------
    omega_w = (w1 - w2).abs() / w1.maximum(w2).clamp(min=eps)
    omega_h = (h1 - h2).abs() / h1.maximum(h2).clamp(min=eps)
    shape_cost = (1 - torch.exp(-omega_w)).pow(SIOU_THETA) + (1 - torch.exp(-omega_h)).pow(SIOU_THETA)

    return iou - 0.5 * (distance_cost + shape_cost)   # 1 - this == Loss_SIoU


def _bbox_loss_forward_siou(
    self,
    pred_dist,
    pred_bboxes,
    anchor_points,
    target_bboxes,
    target_scores,
    target_scores_sum,
    fg_mask,
    imgsz=None,
    stride=None,
):
    """Drop-in replacement for ``BboxLoss.forward`` using SIoU instead of CIoU.

    Mirrors the original ultralytics 8.4.51 implementation exactly; only the IoU
    term is swapped (CIoU -> SIoU). The DFL branch is unchanged.
    """
    from ultralytics.utils.tal import bbox2dist

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    # >>> paper change: SIoU regression loss instead of CIoU <<<
    iou = bbox_siou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)
    loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

    # DFL loss (unchanged from baseline)
    if self.dfl_loss:
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
        loss_dfl = loss_dfl.sum() / target_scores_sum
    else:
        target_ltrb = bbox2dist(anchor_points, target_bboxes)
        target_ltrb = target_ltrb * stride
        target_ltrb[..., 0::2] /= imgsz[1]
        target_ltrb[..., 1::2] /= imgsz[0]
        pred_dist = pred_dist * stride
        pred_dist[..., 0::2] /= imgsz[1]
        pred_dist[..., 1::2] /= imgsz[0]
        loss_dfl = (
            F.l1_loss(pred_dist[fg_mask], target_ltrb[fg_mask], reduction="none").mean(-1, keepdim=True) * weight
        )
        loss_dfl = loss_dfl.sum() / target_scores_sum

    return loss_iou, loss_dfl


def patch_bbox_loss_with_siou() -> None:
    """Monkey-patch ultralytics' BboxLoss to use SIoU. Idempotent."""
    from ultralytics.utils import loss as _loss

    if getattr(_loss.BboxLoss.forward, "_siou_patched", False):
        return
    _bbox_loss_forward_siou._siou_patched = True
    _loss.BboxLoss.forward = _bbox_loss_forward_siou
