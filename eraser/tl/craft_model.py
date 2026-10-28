#!/usr/bin/env python3
"""CRAFT, rebuilt from the checkpoint's own layer shapes.

Character Region Awareness For Text detection predicts two per-pixel maps at half
resolution: a region score, high on each character's body, and an affinity score, high in
the gaps *between* characters of the same word. Only the region score is wanted here --
affinity is what groups characters into words, and grouping is precisely what we do not
want, because it fills the space between glyphs.

The slice boundaries below are read off `craft_mlt_25k.pth` rather than assumed, so the
weights load strictly.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16_bn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + mid_channels, mid_channels, kernel_size=1),
            nn.BatchNorm2d(mid_channels), nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VGG16BN(nn.Module):
    """VGG16-BN cut at the indices the checkpoint uses, plus CRAFT's dilated stage 5."""

    def __init__(self) -> None:
        super().__init__()
        features = list(vgg16_bn(weights=None).features.children())
        # The checkpoint names each layer by its index in the *original* VGG feature list,
        # so slice2 starts at "14", not "0". Rebuilding with Sequential(*features[12:19])
        # renumbers from zero and nothing after slice1 matches.
        self.slice1, self.slice2 = nn.Sequential(), nn.Sequential()
        self.slice3, self.slice4 = nn.Sequential(), nn.Sequential()
        for target, span in ((self.slice1, range(0, 12)), (self.slice2, range(12, 19)),
                             (self.slice3, range(19, 29)), (self.slice4, range(29, 39))):
            for index in span:
                target.add_module(str(index), features[index])
        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1))

    def forward(self, x: torch.Tensor):
        h = self.slice1(x)
        relu2_2 = h
        h = self.slice2(h)
        relu3_2 = h
        h = self.slice3(h)
        relu4_3 = h
        h = self.slice4(h)
        relu5_3 = h
        h = self.slice5(h)
        return h, relu5_3, relu4_3, relu3_2, relu2_2


class CRAFT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.basenet = VGG16BN()
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sources = self.basenet(x)
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        for index, block in ((2, self.upconv2), (3, self.upconv3), (4, self.upconv4)):
            y = F.interpolate(y, size=sources[index].size()[2:], mode="bilinear",
                              align_corners=False)
            y = torch.cat([y, sources[index]], dim=1)
            y = block(y)
        return self.conv_cls(y).permute(0, 2, 3, 1)   # [B, H/2, W/2, 2] -> region, affinity


def load_craft(path: str, device: str = "cuda") -> CRAFT:
    model = CRAFT()
    state = torch.load(path, map_location="cpu", weights_only=True)
    cleaned = {key[len("module."):] if key.startswith("module.") else key: value
               for key, value in state.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"CRAFT weights did not load cleanly: "
                           f"{len(missing)} missing, {len(unexpected)} unexpected")
    return model.to(device).eval()
