#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7
#

import numpy as np
import os
import torch

class HARdataset():
    def __init__(self, path):
        self.dataset_path = path

        self.input_signal_type = [
            "body_acc_x_",
            "body_acc_y_",
            "body_acc_z_",
            "body_gyro_x_",
            "body_gyro_y_",
            "body_gyro_z_",
            "total_acc_x_",
            "total_acc_y_",
            "total_acc_z_"
        ]

        self.labels = [
            "WALKING",
            "WALKING_UPSTAIRS",
            "WALKING_DOWNSTAIRS",
            "SITTING",
            "STANDING",
            "LAYING"
        ]

    def load_X(self, X_signals_paths):
        X_signals = []

        for signal_type_path in X_signals_paths:
            file = open(signal_type_path, 'r')
            # Read dataset from disk, dealing with text files' syntax
            X_signals.append(
                [np.array(serie, dtype=np.float32) for serie in [
                    row.replace('  ', ' ').strip().split(' ') for row in file
                ]]
            )
            file.close()

        return np.transpose(np.array(X_signals), (1, 2, 0))

    def load_y(self, y_path):
        file = open(y_path, 'r')
        # Read dataset from disk, dealing with text file's syntax
        y_ = np.array(
            [elem for elem in [
                row.replace('  ', ' ').strip().split(' ') for row in file
            ]],
            dtype=np.int32
        )
        file.close()
        return y_ - 1

    def load_dataset(self):
        os.path.join(self.dataset_path, 'train')

        X_train_signals_paths = [
            self.dataset_path + "train/Inertial Signals/" + signal + "train.txt" for signal in self.input_signal_type
        ]

        X_test_signals_paths = [
            self.dataset_path + "test/Inertial Signals/" + signal + "test.txt" for signal in self.input_signal_type
        ]

        X_train = self.load_X(X_train_signals_paths)
        X_test = self.load_X(X_test_signals_paths)

        y_train_path = self.dataset_path + "train/y_train.txt"
        y_test_path = self.dataset_path + "test/y_test.txt"

        y_train = self.load_y(y_train_path)
        y_test = self.load_y(y_test_path)

        train_dataset = []
        for i in range(len(X_train)):
            # print(y_train[i])
            # exit()
            train_dataset.append((torch.tensor(X_train[i]), torch.tensor(y_train[i][0], dtype=torch.int64)))

        test_dataset = []
        for i in range(len(X_test)):
            test_dataset.append((torch.tensor(X_test[i]), torch.tensor(y_test[i][0], dtype=torch.int64)))

        return train_dataset, test_dataset
