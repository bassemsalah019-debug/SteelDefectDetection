"""
WIoU (Wise-IoU) bounding-box regression loss.

Reproduces the loss change of LZY-233/yolov8_Imporved-Defect_detection, which
replaces CIoU with WIoU (Tong et al., 2023). Ported faithfully from their
`ultralytics/utils/iou.py` `IoU_Cal` class:

    iou_loss = 1 - IoU                              # their `self.iou`
    dist     = exp( l2_center / l2_box.detach() )   # center dist / enclosing-box diag^2
    WIoU     = dist * iou_loss
    focusing (monotonous, their default True):
        beta = iou_loss.detach() / iou_mean         # iou_mean: momentum running mean
        WIoU *= sqrt(beta)

`iou_mean` is updated with momentum = 1 - 0.5**(1/7000) during training (matching
their constants). Ultralytics 8.4.51 has no WIoU option, so this module provides
`bbox_wiou` and monkey-patches `ultralytics.utils.loss.BboxLoss.forward` to use it
in place of CIoU; the DFL term is untouched. Activated via
`src.modules.register_lzy(wiou=True)` — no edits to the installed package.
"""

import torch
import torch.nn.functional as F


class _WIoUState:
    """Momentum running mean of the IoU-loss term, for WIoU focusing (their IoU_Cal)."""

    iou_mean = 1.0
    momentum = 1.0 - 0.5 ** (1.0 / 7000)
    monotonous = True            # WIoU v1 monotonic focusing (LZY default monotonous=True)


def bbox_wiou(pred: torch.Tensor, target: torch.Tensor, xywh: bool = False,
              eps: float = 1e-7, training: bool = True) -> torch.Tensor:
    """Per-box WIoU loss. Returns shape ``[..., 1]`` (matches Ultralytics' iou term)."""
    if xywh:  # (cx,cy,w,h) -> (x1,y1,x2,y2)
        (px, py, pw, ph) = pred.chunk(4, -1)
        (tx, ty, tw, th) = target.chunk(4, -1)
        pred = torch.cat([px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2], -1)
        target = torch.cat([tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2], -1)

    pred_xy = (pred[..., :2] + pred[..., 2:4]) / 2
    target_xy = (target[..., :2] + target[..., 2:4]) / 2
    pred_wh = pred[..., 2:4] - pred[..., :2]
    target_wh = target[..., 2:4] - target[..., :2]

    min_coord = torch.minimum(pred[..., :4], target[..., :4])
    max_coord = torch.maximum(pred[..., :4], target[..., :4])
    wh_inter = torch.relu(min_coord[..., 2:4] - max_coord[..., :2])
    s_inter = torch.prod(wh_inter, dim=-1)
    s_union = torch.prod(pred_wh, dim=-1) + torch.prod(target_wh, dim=-1) - s_inter + eps
    iou_loss = 1 - s_inter / s_union                       # their `self.iou` (== 1 - IoU)

    l2_center = torch.square(pred_xy - target_xy).sum(dim=-1)
    wh_box = max_coord[..., 2:4] - min_coord[..., :2]        # smallest enclosing box w,h
    l2_box = torch.square(wh_box).sum(dim=-1) + eps
    dist = torch.exp(l2_center / l2_box.detach())            # WIoU distance focusing

    loss = dist * iou_loss
    if _WIoUState.monotonous:                                # WIoU v1 focusing
        beta = iou_loss.detach() / _WIoUState.iou_mean
        loss = loss * beta.clamp(min=eps).sqrt()

    if training:                                             # momentum-update running mean
        with torch.no_grad():
            m = _WIoUState.momentum
            _WIoUState.iou_mean = (1 - m) * _WIoUState.iou_mean + m * float(iou_loss.mean())

    return loss.unsqueeze(-1)


def _bbox_loss_forward_wiou(
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
    """Drop-in replacement for ``BboxLoss.forward`` using WIoU instead of CIoU."""
    from ultralytics.utils.tal import bbox2dist

    weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    # >>> change: WIoU regression loss instead of CIoU <<<
    wiou = bbox_wiou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False,
                     training=torch.is_grad_enabled())
    loss_iou = (wiou * weight).sum() / target_scores_sum

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


def patch_bbox_loss_with_wiou() -> None:
    """Monkey-patch ultralytics' BboxLoss to use WIoU. Idempotent."""
    from ultralytics.utils import loss as _loss

    if getattr(_loss.BboxLoss.forward, "_wiou_patched", False):
        return
    _bbox_loss_forward_wiou._wiou_patched = True
    _loss.BboxLoss.forward = _bbox_loss_forward_wiou
