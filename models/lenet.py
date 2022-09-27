#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7

from torch.nn import Module
from torch import nn

__all__ = ['LeNet']


class LeNet(Module):
    def __init__(self, dataset='mnist', cfg=None, small_classifier=False):
        super(LeNet, self).__init__()

        if cfg is None:
            cfg = [6, 16]

        if dataset != 'mnist':
            exit('lenet only for mnist!')

        self.cfg_last = cfg[-1]
        self.features = nn.Sequential(
            nn.Conv2d(1, cfg[0], 5),
            nn.BatchNorm2d(cfg[0], momentum=None, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(cfg[0], cfg[1], 5),
            nn.BatchNorm2d(cfg[1], momentum=None, track_running_stats=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(4)

        self.classifier1 = nn.Sequential(
            nn.Linear(cfg[-1] * 4 * 4, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(84, 10),
            nn.ReLU()
        )

    def forward(self, x):
        y = self.features(x)
        y = self.avgpool(y)
        y = y.view(y.shape[0], -1)
        x1 = self.classifier1(y)
        y = self.classifier2(x1)
        return y, x1
