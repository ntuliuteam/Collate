#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7


import sys
from torchsummary import summary

sys.path.append("..")

import os
import argparse

import numpy as np
from models.harmodel import HarCNN
import torch
import torch.nn as nn

import tensorflow as tf

from tensorflow.python.framework import graph_util


def feature(x, feature_i):
    for t in range(4):
        cur = t + feature_i * 4
        w_conv = tf.Variable(tf.constant(Conv_weight[cur]))
        b_conv = tf.Variable(tf.constant(Conv_bias[cur]))
        x = tf.nn.conv1d(x, w_conv, stride=1, padding='VALID')
        x = tf.nn.bias_add(x, b_conv)

        if BN_mean[cur] is not None and BN_var[cur] is not None:
            mean = tf.Variable(tf.constant(BN_mean[cur]))
            var = tf.Variable(tf.constant(BN_var[cur]))
        else:
            x_shape = x.get_shape()
            axis = list(range(len(x_shape) - 1))
            mean, var = tf.nn.moments(x, axis)
        scale = tf.Variable(tf.constant(BN_weight[cur]))
        shift = tf.Variable(tf.constant(BN_bias[cur]))
        epsilon = 0.00001
        x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)

        x = tf.nn.relu(x)

        if t == 1 or t == 3:
            x = tf.nn.max_pool1d(x, ksize=2, strides=2, padding='VALID')

    x = tf.transpose(x, perm=[0, 2, 1])
    in_size = last_layer[feature_i] * 26
    x = tf.reshape(x, [-1, in_size])

    w_fc = tf.Variable(tf.constant(Linear_weight[feature_i]))
    b_fc = tf.Variable(tf.constant((Linear_bias[feature_i])))
    x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)
    x = tf.nn.relu(x)

    return x


def classifier(x, classifier_begin):
    w_fc = tf.Variable(tf.constant(Linear_weight[classifier_begin]))
    b_fc = tf.Variable(tf.constant((Linear_bias[classifier_begin])))
    x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)
    x = tf.nn.relu(x)

    w_fc = tf.Variable(tf.constant(Linear_weight[classifier_begin + 1]))
    b_fc = tf.Variable(tf.constant((Linear_bias[classifier_begin + 1])))
    out = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)

    out = tf.nn.relu(out, name='Output')
    return out


class HarCNN_tf:
    def __init__(self, num_classes=6, IMU_num=9):
        self.num_classes = num_classes
        self.IMU_num = IMU_num

        self.features = feature
        self.classifier = classifier

    def forward(self, x):
        out_tmp = []
        for i in range(self.IMU_num):
            out = x[:, :, i]
            out = tf.expand_dims(out, 2)
            out = self.features(out, i)
            out_tmp.append(out)

        out = tf.concat(out_tmp, 1)
        out = self.classifier(out, self.IMU_num)
        return out


def main():
    sess = tf.InteractiveSession()

    x_input = tf.placeholder("float", [None, 128, 9], name="Input")
    lenet_tf = HarCNN_tf()
    x_tmp = lenet_tf.forward(x_input)

    sess.run(tf.global_variables_initializer())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Input', 'Output'])
    with tf.gfile.FastGFile(save, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model", help="The path to saved torch model checkpoint.", required=True)
    parser.add_argument("--save", help="The path where the frozen pb will be saved.", required=True)
    parser.add_argument('--cfg_id', type=int, default=2, help="original model's configuration ID")

    args = parser.parse_args()

    torch_model = args.torch_model
    save = args.save

    cfg_hete = [2, 2, 1, 1, 1]

    cfg_scale = cfg_hete[args.cfg_id]

    model = HarCNN(cfg = [64*cfg_scale] * 36)

    model.cuda()

    checkpoint = torch.load(torch_model)
    best_prec1 = checkpoint['best_prec1']
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    Conv_weight = []
    Conv_bias = []
    BN_weight = []
    BN_bias = []
    BN_mean = []
    BN_var = []
    Linear_weight = []
    Linear_bias = []

    last_layer = []
    last_nonzero = None

    count = 0
    last_nonzeros = []

    ll = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):

            # print( m.weight.data.cpu().numpy().shape)

            if ll < 9:

                process = m.weight.data.cpu().numpy().reshape(512, 64 * cfg_scale, 26)
                process = torch.tensor(process).transpose(0, 1).numpy()

                # print(last_nonzeros[ll])
                process = process[last_nonzeros[ll]]
                process = torch.tensor(process).transpose(0, 1).numpy()
                process = process.reshape(512, last_layer[ll] * 26)
                process = torch.tensor(process).transpose(0, 1).numpy()

                Linear_weight.append(process)
                Linear_bias.append(m.bias.data.cpu().numpy())


            else:

                # print(m.weight.data.transpose(0, 1).cpu().numpy().shape)

                Linear_weight.append(m.weight.data.transpose(0, 1).cpu().numpy())
                Linear_bias.append(m.bias.data.cpu().numpy())
            ll = ll + 1

        if isinstance(m, nn.Conv1d):
            Conv_weight.append(m.weight.data.cpu().numpy())
            Conv_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.BatchNorm1d):
            weight = m.weight.data.cpu().numpy()
            bias = m.bias.data.cpu().numpy()
            nonzero_index = np.union1d(np.nonzero(weight), np.nonzero(bias))
            if len(nonzero_index) == 0:
                nonzero_index = np.array([0])
            BN_weight.append(weight[nonzero_index])
            BN_bias.append(bias[nonzero_index])

            if m.running_mean is not None and m.running_var is not None:

                BN_mean.append(m.running_mean.cpu().numpy()[nonzero_index])
                BN_var.append(m.running_var.cpu().numpy()[nonzero_index])
            else:
                BN_mean.append(None)
                BN_var.append(None)

            conv_last = len(BN_weight) - 1

            if last_nonzero is not None:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 1).numpy()
                process = process[last_nonzero]
                process = torch.tensor(process).transpose(1, 0).transpose(0, 2).numpy()
                Conv_weight[conv_last] = process

                process = Conv_bias[conv_last][nonzero_index]
                Conv_bias[conv_last] = process
            else:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 2).numpy()
                Conv_weight[conv_last] = process
                process = Conv_bias[conv_last][nonzero_index]
                Conv_bias[conv_last] = process

            last_layer_i = len(nonzero_index)
            last_nonzero = nonzero_index

            if count % 4 == 3:
                last_layer.append(last_layer_i)
                last_nonzeros.append(last_nonzero)
                last_nonzero = None

            count = count + 1

    main()
