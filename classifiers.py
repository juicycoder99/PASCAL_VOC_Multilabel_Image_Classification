"""Networks for Assignment 3 (PASCAL VOC multi-label classification).

SimpleNet  - a small from-scratch convolutional network (the warm-up classifier for Part A).
MyNet      - a self-designed compact residual network for Part B (no pre-trained weights).
"""
import torch.nn as nn
import torch.nn.functional as F


class SimpleNet(nn.Module):
    """A small, plain CNN trained from scratch (Part A, experiment 3)."""
    def __init__(self, num_classes=20):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2), # 28
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),# 14
        )
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(128, num_classes))

    def forward(self, x):
        return self.head(self.features(x))


class _ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(out_c)
        self.short = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.short = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                                       nn.BatchNorm2d(out_c))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.short(x))


class MyNet(nn.Module):
    """Self-designed residual network for Part B (no pre-trained weights).

    A compact stem followed by four residual stages with global average pooling and dropout. The
    design borrows the residual idea but is small enough to train from scratch on VOC2007.
    """
    def __init__(self, num_classes=20):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1))                                       # 56x56
        self.stage1 = _ResBlock(64, 64)
        self.stage2 = _ResBlock(64, 128, stride=2)                       # 28
        self.stage3 = _ResBlock(128, 256, stride=2)                      # 14
        self.stage4 = _ResBlock(256, 512, stride=2)                      # 7
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.stem(x)
        x = self.stage4(self.stage3(self.stage2(self.stage1(x))))
        return self.head(x)
