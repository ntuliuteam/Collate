#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7

import torch
import torch.nn as nn
from tensorboard import summary

__all__ = ['HarCNN']


class HarCNN(nn.Module):
    def __init__(self, dataset='har', cfg=None, small_classifier=None, num_classes=6, IMU_num=9):
        super(HarCNN, self).__init__()

        if cfg is None:
            cfg = [64] * (4 * IMU_num)
            # cfg = [i+1 for i in range(36)]
        else:
            assert len(cfg) == (4 * IMU_num)
        self.IMU_num = IMU_num

        # self.IMU_features = []
        # self.classifiers = []

        self.IMU_features = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(1, cfg[i * 4 + 0], 5),
                nn.BatchNorm1d(cfg[i * 4 + 0], momentum=None, track_running_stats=False),
                nn.ReLU(),
                nn.Conv1d(cfg[i * 4 + 0], cfg[i * 4 + 1], 5),
                nn.BatchNorm1d(cfg[i * 4 + 1], momentum=None, track_running_stats=False),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(cfg[i * 4 + 1], cfg[i * 4 + 2], 5),
                nn.BatchNorm1d(cfg[i * 4 + 2], momentum=None, track_running_stats=False),
                nn.ReLU(),
                nn.Conv1d(cfg[i * 4 + 2], cfg[i * 4 + 3], 5),
                nn.BatchNorm1d(cfg[i * 4 + 3], momentum=None, track_running_stats=False),
                nn.ReLU(),
                nn.MaxPool1d(2)) for i in range(IMU_num)

        )

        self.IMU_classifiers = nn.ModuleList(
            nn.Sequential(
                nn.Linear(26 * cfg[i * 4 + 3], 512),
                nn.ReLU(),
            ) for i in range(IMU_num)

        )

        self.classifier1 = nn.Sequential(
            nn.Linear(512 * IMU_num, 512),
            nn.ReLU(),
        )
        self.classifier2 = nn.Sequential(
            nn.Linear(512, num_classes),
        )

    def forward(self, x):

        out_tmp = []
        for i in range(self.IMU_num):
            out = x[:, :, i]
            out = out.unsqueeze(1)
            out = self.IMU_features[i](out)

            out = out.view(out.shape[0], -1)

            out = self.IMU_classifiers[i](out)
            out_tmp.append(out)
        out = torch.cat(out_tmp, 1)
        x1 = self.classifier1(out)
        out = self.classifier2(x1)
        return out, x1
