#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @python: 3.7.7

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import shutil
from torchvision import datasets, transforms


def save_checkpoint(state, is_best, filepath, epoch):
    torch.save(state, os.path.join(filepath, 'checkpoint.pth.tar'))
    if is_best:
        shutil.copyfile(os.path.join(filepath, 'checkpoint.pth.tar'), os.path.join(filepath, 'model_best.pth.tar'))


def print_cfg(net):
    for m2 in net.modules():
        if isinstance(m2, nn.BatchNorm2d) or isinstance(m2, nn.BatchNorm1d):
            nonzero = torch.nonzero(m2.weight.data).size()[0]
            original = m2.weight.data.size()[0]
            print('Before Compact:', original, 'After ZR Compact', nonzero)


def test_save(net, dataset_test, best_prec1, args, epoch, idx=None, save=True):
    acc_test, loss_test = test_img(net, dataset_test, args)
    print(str(idx), "Testing accuracy: {:.2f}".format(acc_test))
    is_best = acc_test > best_prec1
    cur_best = max(acc_test, best_prec1)

    if save:
        if idx is not None:
            path = os.path.join(args.save, str(idx))
            if not os.path.exists(path):
                os.makedirs(path)
        else:
            path = args.save
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net.state_dict(),
            'best_prec1': cur_best,
        }, is_best, filepath=path, epoch=epoch)

    return cur_best


def test_img(net_g, datatest, args):
    net_g.eval()
    test_loss = 0
    correct = 0

    # in the code, we use the BN layer to implement the function of the mask layer. to protect privacy,  we also use
    # the static BN layer without the var/mean from the training data, as used in HeteroFL.
    #
    # without the var/mean from the training data means that the var/mean of the BN layer during inference comes from
    # the batch of test data, and this results in different batch sizes have different var/mean and different 
    # test accuracy. if you want to change the batch size, you should first use the var/mean of all local test 
    # sets to fix the var/mean of BN layers. 
    #
    # the static BN is also used for other methods.
    data_loader = DataLoader(datatest, batch_size=128)
    for idx, (data, target) in enumerate(data_loader):
        if args.gpu != -1:
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            log_probs = net_g(data)

            if isinstance(log_probs, tuple):
                log_probs = log_probs[0]

            # sum up batch loss
            test_loss += F.cross_entropy(log_probs, target, reduction='sum').item()
            # get the index of the max log-probability
            y_pred = log_probs.data.max(1, keepdim=True)[1]
            correct += y_pred.eq(target.data.view_as(y_pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    return accuracy, test_loss

def tranforms_all():
    all_transform = {}

    # 224 is just for larger latency to simulate real application
    # the baseline's accuracy on iid: resolution of 32x32 is 83.18% and 224x224 is 82.72%
    # besides, the transform we used for cifar100 is a resolution of 32x32.
    # this is also used for other methods.
    all_transform['cifar10_train'] = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_transform['cifar10_test'] = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_transform['mnist_train'] = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    all_transform['mnist_test'] = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    all_transform['cifar100_train'] = transforms.Compose([transforms.RandomCrop(32, padding=4),
                                                          transforms.RandomHorizontalFlip(),
                                                          transforms.ToTensor(),
                                                          transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                               std=[0.267, 0.256, 0.276])])
    all_transform['cifar100_test'] = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Normalize(mean=[0.507, 0.487, 0.441],
                                                                              std=[0.267, 0.256, 0.276])])

    # trans_cifar10_train = transforms.Compose([transforms.RandomCrop(32, padding=4),
    #                                           transforms.RandomHorizontalFlip(),
    #                                           transforms.ToTensor(),
    #                                           transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                                std=[0.229, 0.224, 0.225])])
    # trans_cifar10_val = transforms.Compose([transforms.ToTensor(),
    #                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                                              std=[0.229, 0.224, 0.225])])

    return all_transform


def arch_config():
    all_arch = {'cifar10': [64, 192, 384, 256, 256], 'cifar100': [64, 64, 128, 128, 256, 256, 512, 512],
                'mnist': [6, 16], 'har': [64] * 36}

    return all_arch
