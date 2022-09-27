#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7
#

import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import models
import numpy as np

from utils.preprocess_har import HARdataset
from utils.config import args_parser
from utils.functions import tranforms_all, arch_config, test_save

args = args_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()


# in the code, we use the BN layer to implement the function of the mask layer. to protect privacy,  we also use
# the static BN layer without the var/mean from the training data, as used in HeteroFL.


if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

all_tranform = tranforms_all()
all_cfg = arch_config()

if args.dataset == 'cifar10':
    dataset_test = datasets.CIFAR10('./cifar10', train=False, download=True, transform=all_tranform['cifar10_test'])
    cfg_hete = [2, 2, 1, 1, 1]
    for_fc = [4096, 6]
    first_c = 3

    if args.cfg_id == 3:
        for_fc = [1024, 6]

elif args.dataset == 'mnist':
    dataset_test = datasets.MNIST('./mnist/', train=False, download=True, transform=all_tranform['mnist_test'])
    cfg_hete = [2, 2, 1, 1, 1]
    for_fc = [120, 4]
    first_c = 1

elif args.dataset == 'cifar100':
    dataset_test = datasets.CIFAR100('./cifar100/', train=False, download=True, transform=all_tranform['cifar100_test'])
    cfg_hete = [5, 3, 1, 1, 1]
    for_fc = [512, 1]
    first_c = 3

elif args.dataset == 'har':
    Har = HARdataset('./har/')
    dataset_train, dataset_test = Har.load_dataset()
    cfg_hete = [2, 2, 1, 1, 1]
    for_fc = [512, 26]
    first_c = 1

else:
    exit('no support dataset')

assert len(cfg_hete) == args.num_users

hete_cfg = {}
for i in range(args.num_users):
    hete_cfg[i] = [int(j * cfg_hete[i]) for j in all_cfg[args.dataset]]

sc = (args.cfg_id == 3)
model = models.__dict__[args.arch](dataset=args.dataset, cfg=hete_cfg[args.cfg_id],
                                   small_classifier=sc)

if args.cuda:
    model.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_prec1 = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        cur_epoch = checkpoint['epoch']
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, cur_epoch, best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))
else:
    exit('for test')

test_save(model, dataset_test, best_prec1, args, cur_epoch, args.cfg_id, save=False)

total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        size = m.weight.data.shape[0]
        bn[index:(index + size)] = m.weight.data.abs().clone()
        index += size

thre = 0.0
cfg = []
cfg_mask = []

for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
        weight_copy = m.weight.data.clone()
        mask = weight_copy.abs().gt(thre).float().cuda()

        if torch.sum(mask) <= 0:  # guarantee at less one non-zero kernel
            mask[0] = 1.0

        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total kernel: {:d} \t remaining kernel: {:d}'.
              format(k, mask.shape[0], int(torch.sum(mask))))

print('Pre-processing Successful!')

# Make real prune
print(cfg)

newmodel = models.__dict__[args.arch](dataset=args.dataset, cfg=cfg,
                                      small_classifier=sc)
newmodel.cuda()

layer_id_in_cfg = 0
start_mask = torch.ones(first_c)
end_mask = cfg_mask[layer_id_in_cfg]
ll = 0

har_mask = []

for [m0, m1] in zip(model.modules(), newmodel.modules()):
    if isinstance(m0, nn.BatchNorm2d) or isinstance(m1, nn.BatchNorm1d):
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        m1.weight.data = m0.weight.data[idx1].clone()
        m1.bias.data = m0.bias.data[idx1].clone()
        if m0.running_mean is not None and m0.running_var is not None:
            m1.running_mean = m0.running_mean[idx1].clone()
            m1.running_var = m0.running_var[idx1].clone()
        layer_id_in_cfg += 1
        start_mask = end_mask.clone()
        if layer_id_in_cfg < len(cfg_mask):
            end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.Conv2d):

        if 'resnet18' in args.arch and layer_id_in_cfg in [7, 12, 17]:
            m1.weight.data = m0.weight.data.clone()
            if hasattr(m0, 'bias') and m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()

        else:
            if len(start_mask) == 1:
                w = m0.weight.data.clone()
            else:
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                w = m0.weight.data[:, idx0, :, :].clone()

            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            w = w[idx1, :, :, :].clone()
            m1.weight.data = w.clone()
            if hasattr(m0, 'bias') and m0.bias is not None:
                m1.bias.data = m0.bias.data[idx1].clone()

    elif isinstance(m0, nn.Conv1d):
        if layer_id_in_cfg % 4 == 0:
            if layer_id_in_cfg != 0:
                har_mask.append(start_mask)
            start_mask = torch.ones(first_c)
            end_mask = cfg_mask[layer_id_in_cfg]

        if len(start_mask) == 1:
            w = m0.weight.data.clone()
        else:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            w = m0.weight.data[:, idx0, :].clone()

            if len(w.shape) < 3:
                w = w.reshape(w.shape[0], 1, w.shape[1])

        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        w = w[idx1, :, :].clone()
        if len(w.shape) < 3:
            w = w.reshape(1, w.shape[0], w.shape[1])

        m1.weight.data = w.clone()
        if hasattr(m0, 'bias') and m0.bias is not None:
            m1.bias.data = m0.bias.data[idx1].clone()
            if len(m1.bias.data.shape) < 1:
                m1.bias.data = m1.bias.data.reshape(1)

    elif isinstance(m0, nn.Linear):
        if 'resnet' in args.arch:
            m1.weight.data = m0.weight.data.clone()

            if hasattr(m0, 'bias') and m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()

        elif 'HarCNN' in args.arch:
            if ll < 9:
                final_fc = hete_cfg[args.cfg_id][ll * 4 + 3]
                tmp = m0.weight.data.reshape((for_fc[0], final_fc, for_fc[1]))
                if ll == 8:
                    idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                else:
                    idx0 = np.squeeze(np.argwhere(np.asarray(har_mask[ll].cpu().numpy())))
                m1.weight.data = tmp[:, idx0, :].reshape((for_fc[0], len(idx0) * for_fc[1])).clone()
            else:
                m1.weight.data = m0.weight.data.clone()
            if hasattr(m0, 'bias') and m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()

        else:
            if ll == 0:
                final_fc = hete_cfg[args.cfg_id][-1]
                tmp = m0.weight.data.reshape((for_fc[0], final_fc, for_fc[1], for_fc[1]))
                idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
                m1.weight.data = tmp[:, idx0, :, :].reshape((for_fc[0], len(idx0) * for_fc[1] * for_fc[1])).clone()

            else:
                m1.weight.data = m0.weight.data.clone()
            if hasattr(m0, 'bias') and m0.bias is not None:
                m1.bias.data = m0.bias.data.clone()

        ll = ll + 1

test_save(newmodel, dataset_test, 0.0, args, cur_epoch, args.cfg_id)
