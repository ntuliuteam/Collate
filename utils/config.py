#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7

import argparse


def args_parser():
    parser = argparse.ArgumentParser(description='PyTorch Implement Of Collate (Review Version)')
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training')
    parser.add_argument('--test_bs', type=int, default=128, help="test batch size")
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--save', default='./logs', type=str, metavar='PATH', help='path to save the model')
    parser.add_argument('--arch', default='test', type=str, help='model architecture to use')
    parser.add_argument('--iid', action='store_true', help='whether i.i.d or not')
    parser.add_argument('--num_users', type=int, default=5, help="number of users: K")
    parser.add_argument('--gpu', type=int, default=0, help="GPU ID, -1 for CPU")
    parser.add_argument('--cfg_id', type=int, default=2, help="original model's configuration ID")
    parser.add_argument('--print_cfg', action='store_true', help="print the left kernels in each layer")

    args = parser.parse_args()
    return args
