#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7

import torch.nn as nn
import torch

'''
modified to fit dataset size
'''

__all__ = ['AlexNet']


class AlexNet(nn.Module):
    def __init__(self, dataset='cifar10', cfg=None, small_classifier=False):
        super(AlexNet, self).__init__()

        if cfg is None:
            cfg = [64, 192, 384, 256, 256]

        self.cfg_last = cfg[-1]

        if dataset == 'cifar10':
            num_classes = 10
        elif dataset == 'cifar100':
            num_classes = 100

        self.features = nn.Sequential(
            nn.Conv2d(3, cfg[0], kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(cfg[0], momentum=None, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(cfg[0], cfg[1], kernel_size=5, padding=2),
            nn.BatchNorm2d(cfg[1], momentum=None, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(cfg[1], cfg[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[2], momentum=None, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg[2], cfg[3], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[3], momentum=None, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg[3], cfg[4], kernel_size=3, padding=1),
            nn.BatchNorm2d(cfg[4], momentum=None, track_running_stats=False),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(6)

        if small_classifier:
            self.classifier0 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(cfg[4] * 6 * 6, 1024),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(1024, 1024),
                nn.ReLU(inplace=True),
            )
        else:
            self.classifier0 = nn.Sequential(
                nn.Dropout(),
                nn.Linear(cfg[4] * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 1024),
                nn.ReLU(inplace=True),
            )

        self.classifier1 = nn.Sequential(
            nn.Linear(1024, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), self.cfg_last * 6 * 6)
        x1 = self.classifier0(x)
        x = self.classifier1(x1)
        return x, x1
