"""
MPCA — MultiPath Coordinate Attention.

Reproduces the attention module from:
  "A lightweight algorithm for steel surface defect detection using improved
   YOLOv8", Scientific Reports (2025), s41598-025-93469-5.

Paper (Methods): MPCA is inserted "at the end of the backbone network". The
module splits processing into FOUR parallel paths; each path encodes spatial
position with a horizontal (along-W) and a vertical (along-H) global-average
pooling — the Coordinate-Attention scheme (Hou et al., CVPR 2021) — then the
direction-encoded vectors pass through a 1x1 conv / MLP to produce channel-wise
attention weights. The weighted maps from the four paths are combined by
addition.

This is a faithful, paper-literal reconstruction (the paper does not publish
MPCA's exact channel/kernel sizes or release code):
  - [PAPER]   four parallel paths; horizontal + vertical GAP per path;
              1x1 conv / MLP -> channel weights; channel-wise multiply;
              paths combined by ADDITION.
  - [DEFAULT] reduction ratio r=32 (the Coordinate-Attention default).
No extra tricks are added (e.g. no input residual) so the module matches the
paper description exactly.

CHANNEL-PRESERVING (out_channels == in_channels). Sub-layers build lazily on the
first forward, so it slots into Ultralytics' `parse_model` `else` branch (which
sets c2 = ch[f]) with an empty YAML arg list `[-1, 1, MPCA, []]` — no edits to
the installed ultralytics package are needed.

Pooling uses `mean()` (not AdaptiveAvgPool2d) so the backward pass is
deterministic under Ultralytics' seeded/deterministic training mode.
"""

import torch
import torch.nn as nn


class _CoordAttPath(nn.Module):
    """One Coordinate-Attention path: H/V mean-pooling -> shared 1x1 -> two 1x1 -> sigmoid gates."""

    def __init__(self, channels: int, reduction: int = 32):
        super().__init__()
        mip = max(8, channels // reduction)        # reduced (mid) channels
        # Shared transform on the concatenated direction-encoded vector ("the MLP").
        self.conv1 = nn.Conv2d(channels, mip, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()                  # h-swish, as in Coordinate Attention
        # Per-direction expansion back to C, then sigmoid -> attention gates.
        self.conv_h = nn.Conv2d(mip, channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.shape
        x_h = x.mean(dim=3, keepdim=True)              # [B,C,H,1] pool along width
        x_w = x.mean(dim=2, keepdim=True).permute(0, 1, 3, 2)  # [B,C,W,1] pool along height
        y = torch.cat([x_h, x_w], dim=2)               # [B,C,H+W,1]
        y = self.act(self.bn1(self.conv1(y)))          # shared reduce + h-swish
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)                  # back to [B,C,1,W]
        a_h = torch.sigmoid(self.conv_h(x_h))          # [B,C,H,1]
        a_w = torch.sigmoid(self.conv_w(x_w))          # [B,C,1,W]
        return x * a_h * a_w                            # coordinate-weighted feature


class MPCA(nn.Module):
    """MultiPath Coordinate Attention (paper: end-of-backbone attention).

    Args:
        c1 (int | None): input channels. Left as None when built from a YAML with
            empty args; inferred lazily on the first forward (Ultralytics' `else`
            branch makes the layer channel-preserving regardless).
        reduction (int): channel reduction ratio per path (default 32).
        n_paths (int): number of parallel coordinate-attention paths (paper: 4).
    """

    def __init__(self, c1: int | None = None, reduction: int = 32, n_paths: int = 4):
        super().__init__()
        self.reduction = reduction
        self.n_paths = n_paths
        self.c1 = c1
        self.paths: nn.ModuleList | None = None
        self.fuse: nn.Conv2d | None = None
        if c1 is not None:
            self._build(c1)

    def _build(self, channels: int) -> None:
        # Four independent coordinate-attention paths (paper: "four paths").
        self.paths = nn.ModuleList(
            _CoordAttPath(channels, self.reduction) for _ in range(self.n_paths)
        )
        # 1x1 conv that combines the added paths (the "MLP" combination step).
        self.fuse = nn.Conv2d(channels, channels, kernel_size=1)
        self.c1 = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.paths is None:                     # lazy build on first forward
            self._build(x.shape[1])
            self.to(x.device, x.dtype)
        out = sum(path(x) for path in self.paths)  # combine paths by ADDITION (paper)
        return self.fuse(out)                      # 1x1 conv combination
