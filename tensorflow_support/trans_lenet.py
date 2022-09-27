#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7


import sys
from torchsummary import summary

sys.path.append("..")

import os
import argparse

import numpy as np
from models.lenet import LeNet
import torch
import torch.nn as nn

import tensorflow as tf

from tensorflow.python.framework import graph_util


def feature(x):
    cur = 0
    w_conv = tf.Variable(tf.constant(Conv_weight[cur]))
    b_conv = tf.Variable(tf.constant(Conv_bias[cur]))
    x = tf.nn.conv2d(x, w_conv, strides=1, padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
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

    x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    cur = 1
    w_conv = tf.Variable(tf.constant(Conv_weight[cur]))
    b_conv = tf.Variable(tf.constant(Conv_bias[cur]))
    x = tf.nn.conv2d(x, w_conv, padding=[[0, 0], [0, 0], [0, 0], [0, 0]])
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

    x = tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    return x


def classifier(x):
    x = tf.transpose(x, perm=[0, 3, 1, 2])
    in_size = last_layer * 4 * 4
    x = tf.reshape(x, [-1, in_size])
    w_fc = tf.Variable(tf.constant(Linear_weight[0]))
    b_fc = tf.Variable(tf.constant((Linear_bias[0])))
    x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)
    x = tf.nn.relu(x)

    w_fc = tf.Variable(tf.constant(Linear_weight[1]))
    b_fc = tf.Variable(tf.constant((Linear_bias[1])))
    x = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)
    x = tf.nn.relu(x)

    w_fc = tf.Variable(tf.constant(Linear_weight[2]))
    b_fc = tf.Variable(tf.constant((Linear_bias[2])))
    out = tf.nn.bias_add(tf.matmul(x, w_fc), b_fc)

    out = tf.nn.relu(out, name='Output')
    return out


class LeNet_tf:
    def __init__(self, num_classes=100):
        self.num_classes = num_classes

        self.features = feature
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def main():
    sess = tf.InteractiveSession()

    x_input = tf.placeholder("float", [None, 28, 28, 1], name="Input")
    lenet_tf = LeNet_tf()
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
    small = False

    model = LeNet(cfg=[6 * cfg_scale, 16 * cfg_scale], small_classifier=small)

    model.cuda()

    checkpoint = torch.load(torch_model)
    best_prec1 = checkpoint['best_prec1']
    print(best_prec1)
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

    last_layer = 0
    last_nonzero = None

    ll = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if ll == 0:


                process = m.weight.data.cpu().numpy().reshape(120, 16*cfg_scale, 4, 4)
                process = torch.tensor(process).transpose(0, 1).numpy()
                process = process[last_nonzero]
                process = torch.tensor(process).transpose(0, 1).numpy()
                process = process.reshape(120, last_layer * 4 * 4)
                process = torch.tensor(process).transpose(0, 1).numpy()

                Linear_weight.append(process)
                Linear_bias.append(m.bias.data.cpu().numpy())

                ll = ll + 1
            else:
                Linear_weight.append(m.weight.data.transpose(0, 1).cpu().numpy())
                Linear_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.Conv2d):
            Conv_weight.append(m.weight.data.cpu().numpy())
            Conv_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.BatchNorm2d):
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
                process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

                process = Conv_bias[conv_last][nonzero_index]
                Conv_bias[conv_last] = process
            else:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

                process = Conv_bias[conv_last][nonzero_index]
                Conv_bias[conv_last] = process

            last_layer = len(nonzero_index)
            last_nonzero = nonzero_index

    main()
