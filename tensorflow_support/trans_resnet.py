#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7


import argparse
import sys

sys.path.append("..")
import numpy as np
from models import resnet18 as resnet18_torch
import torch
import torch.nn as nn
import tensorflow as tf
from tensorflow.python.framework import graph_util


class Conv2D:
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=7, padding=None, bias=False):
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.bias = bias

    def forward(self, x):
        global cur_conv
        if self.bias:
            print('no bias')
            exit()
        else:
            w_conv = tf.Variable(tf.constant(Conv_weight[cur_conv]))
            x = tf.nn.conv2d(x, w_conv, strides=[1, self.stride, self.stride, 1], padding=self.padding)
            cur_conv = cur_conv + 1
            return x


class BN:
    def __init__(self):
        pass

    def forward(self, x):
        # size = self.size
        global cur_bn

        if BN_mean[cur_bn] is not None and BN_var[cur_bn] is not None:
            mean = tf.Variable(tf.constant(BN_mean[cur_bn]))
            var = tf.Variable(tf.constant(BN_var[cur_bn]))
        else:
            x_shape = x.get_shape()
            axis = list(range(len(x_shape) - 1))
            mean, var = tf.nn.moments(x, axis)

        # mean = tf.Variable(tf.constant(BN_mean[cur_bn]))
        # var = tf.Variable(tf.constant(BN_var[cur_bn]))
        scale = tf.Variable(tf.constant(BN_weight[cur_bn]))
        shift = tf.Variable(tf.constant(BN_bias[cur_bn]))
        epsilon = 1e-5
        cur_bn = cur_bn + 1
        x = tf.nn.batch_normalization(x, mean, var, shift, scale, epsilon)
        return x


class Sequential:
    def __init__(self, *args):
        self.args = args

    def forward(self, x):
        output = x
        for lay in self.args:
            output = lay.forward(output)
        return output


class BasicBlock:
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, out_channels_tmp=None):
        if out_channels_tmp is None:
            out_channels_tmp = out_channels

        self.conv1 = Conv2D(in_channels, out_channels_tmp, kernel_size=3, stride=stride,
                            padding=[[0, 0], [1, 1], [1, 1], [0, 0]], bias=False)
        self.bn1 = BN()
        self.conv2 = Conv2D(out_channels_tmp, out_channels * BasicBlock.expansion, kernel_size=3, stride=1,
                            padding=[[0, 0], [1, 1], [1, 1], [0, 0]], bias=False)
        self.bn2 = BN()

        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = Sequential(
                Conv2D(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride,
                       padding=[[0, 0], [0, 0], [0, 0], [0, 0]], bias=False),
                BN())

        else:
            self.shortcut = Sequential()

    def forward(self, x):
        x1 = self.conv1.forward(x)
        x1 = self.bn1.forward(x1)
        x1 = tf.nn.relu(x1)

        x1 = self.conv2.forward(x1)
        x1 = self.bn2.forward(x1)

        x2 = self.shortcut.forward(x)

        out = x1 + x2

        out = tf.nn.relu(out)

        return out


class ResNet:
    def __init__(self, block, num_block, num_classes=100, cfg=None):

        self.in_channels = 64

        if cfg is None:
            cfg = [None] * 100

        self.conv1 = Sequential(
            Conv2D(in_planes=3, out_planes=64, stride=1, kernel_size=3, padding=[[0, 0], [1, 1], [1, 1], [0, 0]],
                   bias=False),
            BN())

        self.conv2_x = self._make_layer(block, 64, num_block[0], 1, out_channels_tmp=cfg[0:num_block[0]])

        self.conv3_x = self._make_layer(block, 128, num_block[1], 2,
                                        out_channels_tmp=cfg[num_block[0]:num_block[0] + num_block[1]])
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2, out_channels_tmp=cfg[num_block[0] + num_block[1]:
                                                                                          num_block[0] + num_block[1] +
                                                                                          num_block[2]])
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2, out_channels_tmp=cfg[num_block[0] + num_block[1] +
                                                                                          num_block[2]:num_block[0] +
                                                                                                       num_block[1] +
                                                                                                       num_block[2] +
                                                                                                       num_block[3]])

        # self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride, out_channels_tmp=None):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride_i in range(len(strides)):
            layers.append(block(self.in_channels, out_channels, strides[stride_i], out_channels_tmp[stride_i]))
            self.in_channels = out_channels * block.expansion

        return Sequential(*layers)

    def forward(self, x):
        output = self.conv1.forward(x)
        output = tf.nn.relu(output)
        output = self.conv2_x.forward(output)
        output = self.conv3_x.forward(output)
        output = self.conv4_x.forward(output)
        output = self.conv5_x.forward(output)

        print(output.shape)

        output = tf.nn.avg_pool2d(output, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')

        print(output.shape)

        output = tf.transpose(output, perm=[0, 3, 1, 2])
        in_size = last_layer
        output = tf.reshape(output, [-1, in_size])

        w_fc = tf.Variable(tf.constant(Linear_weight[0]))
        b_fc = tf.Variable(tf.constant((Linear_bias[0])))
        output = tf.nn.bias_add(tf.matmul(output, w_fc), b_fc, name="Output")

        return output


def resnet18():

    # the cfg is not used in pb models. just for the same formate of the torch model
    cfg = [64, 64, 64, 64, 64, 128, 128, 128, 128, 128, 256, 256, 256, 256, 256, 512, 512, 512, 512, 512]

    no_skip = [1, 3, 5, 8, 10, 13, 15, 18]
    cfg_tmp = []
    for i in no_skip:
        cfg_tmp.append(cfg[i])

    return ResNet(BasicBlock, [2, 2, 2, 2], cfg=cfg_tmp)


def main():
    sess = tf.InteractiveSession()

    x_input = tf.placeholder("float", [None, 32, 32, 3], name="Input")

    model = resnet18()
    out = model.forward(x_input)
    sess.run(tf.global_variables_initializer())

    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph_def, ['Input', 'Output'])
    with tf.gfile.FastGFile(save, mode='wb') as f:
        f.write(constant_graph.SerializeToString())


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--torch_model", help="The path to a frozen model file.", required=True)
    parser.add_argument("--save", help="The name of the dataset to load.", required=True)
    parser.add_argument('--cfg_id', type=int, default=2, help="original model's configuration ID")

    args = parser.parse_args()

    torch_model = args.torch_model
    save = args.save
    cfg_hete = [5, 3, 1, 1, 1]

    cfg_scale = cfg_hete[args.cfg_id]

    model = resnet18_torch(cfg=[64 * cfg_scale, 64 * cfg_scale, 128 * cfg_scale, 128 * cfg_scale, 256 * cfg_scale,
            256 * cfg_scale, 512 * cfg_scale, 512 * cfg_scale])


    model = model.cuda()

    checkpoint = torch.load(torch_model)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    Conv_weight = []
    # Conv_bias = []
    BN_weight = []
    BN_bias = []
    BN_mean = []
    BN_var = []
    Linear_weight = []
    Linear_bias = []

    last_layer = 0
    last_nonzero = None

    cur_bn = 0
    cur_conv = 0
    skip_list = [0, 2, 4, 6, 7, 9, 11, 12, 14, 16, 17, 19]
    # no_skip = [1, 3, 5, 8, 10, 13, 15, 18]

    skip = 0
    # first_nonezero = None

    for m in model.modules():

        if isinstance(m, nn.Linear):
            Linear_weight.append(m.weight.data.cpu().transpose(0, 1).numpy())
            Linear_bias.append(m.bias.data.cpu().numpy())

        if isinstance(m, nn.Conv2d):
            Conv_weight.append(m.weight.data.cpu().numpy())

        if isinstance(m, nn.BatchNorm2d):

            weight = m.weight.data.cpu().numpy()
            bias = m.bias.data.cpu().numpy()
            nonzero_index = np.union1d(np.nonzero(weight), np.nonzero(bias))

            if len(nonzero_index) == 0:
                nonzero_index = np.array([0])

            if m.running_mean is not None and m.running_var is not None:

                BN_mean.append(m.running_mean.cpu().numpy()[nonzero_index])
                BN_var.append(m.running_var.cpu().numpy()[nonzero_index])
            else:
                BN_mean.append(None)
                BN_var.append(None)

            BN_weight.append(weight[nonzero_index])
            BN_bias.append(bias[nonzero_index])

            conv_last = len(BN_weight) - 1

            if last_nonzero is not None:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 1).numpy()
                process = process[last_nonzero]
                process = torch.tensor(process).transpose(1, 0).transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

            else:
                process = Conv_weight[conv_last][nonzero_index]
                process = torch.tensor(process)
                process = process.transpose(0, 3).transpose(1, 2).transpose(0, 1).numpy()
                Conv_weight[conv_last] = process

            last_layer = len(nonzero_index)
            last_nonzero = nonzero_index
            if skip in skip_list:
                last_nonzero = None
            skip += 1

    main()
