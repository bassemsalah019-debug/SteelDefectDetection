"""
ResBlock_CBAM — residual bottleneck block with CBAM attention.

Reproduces modification of the GitHub project
  LZY-233/yolov8_Imporved-Defect_detection
(steel-defect "improved YOLOv8": GhostConv backbone + ResBlock_CBAM in the head +
WIoU loss; reported mAP@0.5 = 0.792 on NEU-DET).

ResBlock_CBAM (their `ultralytics/nn/modules/conv.py`):
  a 1x1 -> 3x3 -> 1x1 bottleneck (BN + LeakyReLU), then CBAM (Woo et al. 2018:
  channel attention then spatial attention), then a residual add and ReLU.
In their YAML it is inserted after each neck C2f (4 places); Detect reads the
three CBAM outputs (P3/P4/P5).

This port is CHANNEL-PRESERVING (out == in, expansion=1, no downsampling — which
is how their config uses it: in_places == places). Sub-layers build lazily on the
first forward so the block slots into Ultralytics' `parse_model` `else` branch
with empty YAML args `[-1, 1, ResBlock_CBAM, []]` (custom-module args are NOT
width-scaled, so inferring channels at runtime is the robust choice — same pattern
as MPCA). Channel pooling uses `mean()` (not AdaptiveAvgPool2d) so the backward is
deterministic under Ultralytics' `deterministic=True` training mode.
"""

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """CBAM channel attention: global-avg-pool -> 1x1 -> sigmoid gate."""

    def __init__(self, channels: int):
        super().__init__()
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # mean over H,W (deterministic) instead of AdaptiveAvgPool2d for det. backward
        return x * self.act(self.fc(x.mean(dim=(2, 3), keepdim=True)))


class SpatialAttention(nn.Module):
    """CBAM spatial attention: [avg,max]-over-channels -> conv -> sigmoid gate."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.mean(1, keepdim=True), x.max(1, keepdim=True)[0]], dim=1)
        return x * self.act(self.cv1(pooled))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (channel then spatial)."""

    def __init__(self, c1: int, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial_attention(self.channel_attention(x))


class ResBlock_CBAM(nn.Module):
    """Residual bottleneck + CBAM (channel-preserving, lazy-built).

    Args:
        c1 (int | None): input channels. Left None when built from a YAML with
            empty args; inferred on the first forward. The block is
            channel-preserving (places == in_places, expansion == 1).
        expansion (int): bottleneck output multiplier (paper-faithful default 1).
    """

    def __init__(self, c1: int | None = None, expansion: int = 1):
        super().__init__()
        self.expansion = expansion
        self.c1 = c1
        self.bottleneck: nn.Sequential | None = None
        self.cbam: CBAM | None = None
        self.relu = nn.ReLU(inplace=True)
        if c1 is not None:
            self._build(c1)

    def _build(self, c: int) -> None:
        p = c  # places == in_places (channel-preserving, downsampling=False)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(c, p, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(p),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(p, p, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(p),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(p, p * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(p * self.expansion),
        )
        self.cbam = CBAM(p * self.expansion)
        self.c1 = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bottleneck is None:                 # lazy build on first forward
            self._build(x.shape[1])
            self.to(x.device, x.dtype)
        out = self.cbam(self.bottleneck(x))
        out = out + x                                # residual (in == out)
        return self.relu(out)
