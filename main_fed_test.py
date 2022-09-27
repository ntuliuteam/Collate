#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7
#
from __future__ import print_function

from torchvision import datasets, transforms
import torch
import torch.nn as nn
from utils.sampling import cifar_iid, cifar_noniid, mnist_noniid, mnist_iid, har_iid, har_noniid
from utils.preprocess_har import HARdataset
from utils.functions import test_save, tranforms_all, arch_config, print_cfg
from utils.config import args_parser
import os
import models

args = args_parser()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

all_tranform = tranforms_all()
all_cfg = arch_config()

# Non-IID default setting is used for continue training from Checkpoint
if args.dataset == 'cifar10':

    dataset_train = datasets.CIFAR10('./cifar10', train=True, download=True, transform=all_tranform['cifar10_train'])
    dataset_test = datasets.CIFAR10('./cifar10', train=False, download=True, transform=all_tranform['cifar10_test'])

    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
        label_split = [None] * args.num_users
    else:
        dict_users, label_split = cifar_noniid(dataset_train, args.num_users)

    cfg_hete = [2, 2, 1, 1, 1]


elif args.dataset == 'mnist':

    dataset_train = datasets.MNIST('./mnist/', train=True, download=True, transform=all_tranform['mnist_train'])
    dataset_test = datasets.MNIST('./mnist/', train=False, download=True, transform=all_tranform['mnist_test'])

    if args.iid:
        dict_users = mnist_iid(dataset_train, args.num_users)
        label_split = [None] * args.num_users
    else:
        dict_users, label_split = mnist_noniid(dataset_train, args.num_users, class_on_device=2)

    cfg_hete = [2, 2, 1, 1, 1]


elif args.dataset == 'cifar100':

    dataset_train = datasets.CIFAR100('./cifar100/', train=True, download=True, transform=all_tranform['cifar100_test'])
    dataset_test = datasets.CIFAR100('./cifar100/', train=False, download=True, transform=all_tranform['cifar100_test'])

    if args.iid:
        dict_users = cifar_iid(dataset_train, args.num_users)
        label_split = [None] * args.num_users

    else:
        dict_users, label_split = cifar_noniid(dataset_train, args.num_users, cifar10=False, class_on_device=2)

    cfg_hete = [5, 3, 1, 1, 1]


elif args.dataset == 'har':

    Har = HARdataset('./har/')
    dataset_train, dataset_test = Har.load_dataset()

    if args.iid:
        dict_users = har_iid(dataset_train, args.num_users)
        label_split = [None] * args.num_users
    else:
        dict_users, label_split = har_noniid(dataset_train, args.num_users, class_on_device=2)

    cfg_hete = [2, 2, 1, 1, 1]

else:
    exit('no support dataset')

assert len(cfg_hete) == args.num_users

hete_cfg = {}
for i in range(args.num_users):
    hete_cfg[i] = [int(j * cfg_hete[i]) for j in all_cfg[args.dataset]]

sc = (args.cfg_id == 3)
net_tmp = models.__dict__[args.arch](dataset=args.dataset, cfg=hete_cfg[args.cfg_id],
                                     small_classifier=sc)

if args.cuda:
    net_tmp.cuda()

if args.resume:
    if os.path.isfile(args.resume):
        print("=> loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        best_prec1 = checkpoint['best_prec1']
        net_tmp.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {}) Prec1: {:f}"
              .format(args.resume, checkpoint['epoch'], best_prec1))
    else:
        print("=> no checkpoint found at '{}'".format(args.resume))

    test_save(net_tmp, dataset_test, 0.0, args, 0, args.cfg_id, save=False)

    if args.print_cfg:
        print_cfg(net_tmp)

else:
    exit('for test')
