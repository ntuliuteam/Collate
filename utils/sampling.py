#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7


import numpy as np
# from torchvision import datasets, transforms
# Non-IID default setting is used for continue training from Checkpoint

def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users=5, label_split=None, default=True, class_on_device=2):
    labels = dataset.train_labels.numpy()
    if default:

        if class_on_device == 2 and num_users == 5:
            label_split = {0: [0, 1, 5, 4], 1: [2, 3, 6, 5], 2: [3, 9, 4, 7], 3: [0, 8, 9, 1], 4: [2, 7, 6, 8]}
        else:
            exit('no default')
    else:
        exit('change the default setting for other conditions')

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_imgs = len(labels)
    idxs = np.arange(all_imgs)
    idxs_labels = np.vstack((idxs, labels))

    mask = np.unique(labels)
    num_per_label = {}
    for v in mask:
        num_per_label[v] = np.sum(labels == v)
    tmp = []
    for i in label_split.keys():
        tmp += label_split[i]
    tmp = np.array(tmp)
    mask = np.unique(tmp)
    num_per_class = {}
    for v in mask:
        num_per_class[v] = np.sum(tmp == v)

    # exit()
    label_class = {}

    for i in range(num_users):
        label_class[i] = {}
        for j in label_split[i]:
            label_class[i][j] = 0

    for img_i in range(all_imgs):
        for user_i in range(num_users):
            label_i = idxs_labels[1, img_i]
            total_num_label_i = num_per_label[label_i]
            label_used = num_per_class[label_i]
            if label_i in label_split[user_i] and label_class[user_i][label_i] <= total_num_label_i / label_used:
                dict_users[user_i] = np.append(dict_users[user_i], idxs_labels[0, img_i])
                label_class[user_i][label_i] += 1
                break

    return dict_users, label_split


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users=5, label_split=None, default=True, cifar10=True,
                 class_on_device=2):
    labels = np.array(dataset.targets)

    if default:
        if cifar10:
            if class_on_device == 2 and num_users == 5:
                label_split = {0: [3, 5, 0, 7], 1: [7, 2, 3, 1], 2: [8, 9, 6, 0], 3: [4, 2, 1, 8], 4: [6, 4, 5, 9]}
            else:
                exit('no default')
        else:

            if class_on_device == 2 and num_users == 5:
                label_split = {0: [92, 73, 52, 72, 99, 65, 0, 75, 41, 54, 88, 94, 56, 78, 9, 4, 27,
                                   58, 44, 1, 55, 46, 23, 51, 43, 59, 14, 66, 80, 36, 30, 60, 85, 34,
                                   31, 24, 8, 67, 47, 62],
                               1: [25, 51, 23, 6, 19, 91, 32, 13, 55, 69, 10, 46, 43, 40, 22, 86, 21,
                                   53, 3, 89, 52, 4, 27, 99, 56, 16, 97, 90, 45, 39, 82, 81, 79, 11,
                                   87, 71, 26, 96, 17, 57],
                               2: [77, 7, 97, 42, 16, 90, 48, 66, 64, 36, 59, 18, 39, 50, 84, 93, 80,
                                   14, 45, 95, 54, 78, 94, 92, 0, 86, 40, 89, 13, 21, 76, 33, 68, 5,
                                   83, 2, 63, 98, 70, 12],
                               3: [31, 82, 87, 79, 33, 81, 74, 68, 76, 34, 5, 38, 35, 30, 29, 83, 28,
                                   85, 60, 11, 72, 65, 58, 44, 73, 69, 32, 22, 19, 91, 18, 48, 50, 77,
                                   93, 37, 15, 49, 61, 20],
                               4: [71, 17, 63, 12, 62, 8, 96, 37, 2, 98, 57, 61, 20, 47, 15, 26, 49,
                                   24, 67, 70, 75, 88, 41, 1, 9, 6, 3, 25, 53, 10, 7, 42, 64, 95,
                                   84, 29, 74, 38, 28, 35]}

            else:
                exit('no default')
    else:
        if label_split is not None:
            assert len(label_split.keys()) == num_users
        else:
            label_split = {}

            all_label = np.array(list(set(labels)))
            np.random.shuffle(all_label)
            tmp = np.split(all_label, 5)
            len_label = len(tmp[0])
            for img_i in range(num_users):
                label_split[img_i] = tmp[img_i]
            not_choice = []
            count = []
            for _ in all_label:
                count.append(0)

            for c_i in range(1, class_on_device):
                left_label = set(np.repeat(all_label, 1))

                for img_i in range(num_users):
                    for img_j in range(num_users):
                        if img_j != img_i:

                            choice_list = (left_label - set(not_choice) - set(label_split[img_i])) & set(
                                label_split[img_j])
                            choice_list = np.array(list(choice_list))

                            tmp = np.random.choice(choice_list, int(len_label / (num_users - 1)), replace=False)

                            left_label = left_label - set(tmp)

                            for tmp_i in tmp:
                                count[tmp_i] += 1
                                if count[tmp_i] == class_on_device:
                                    not_choice.append(tmp_i)

                            label_split[img_i] = np.concatenate((label_split[img_i], tmp), axis=0)

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_imgs = len(labels)
    idxs = np.arange(all_imgs)
    idxs_labels = np.vstack((idxs, labels))

    mask = np.unique(labels)
    num_per_label = {}
    for v in mask:
        num_per_label[v] = np.sum(labels == v)
    tmp = []
    for i in label_split.keys():
        tmp += label_split[i]
    tmp = np.array(tmp)
    mask = np.unique(tmp)
    num_per_class = {}
    for v in mask:
        num_per_class[v] = np.sum(tmp == v)

    # exit()
    label_class = {}

    for i in range(num_users):
        label_class[i] = {}
        for j in label_split[i]:
            label_class[i][j] = 0

    for img_i in range(all_imgs):
        for user_i in range(num_users):
            label_i = idxs_labels[1, img_i]
            total_num_label_i = num_per_label[label_i]
            label_used = num_per_class[label_i]
            if label_i in label_split[user_i] and label_class[user_i][label_i] <= total_num_label_i / label_used:
                dict_users[user_i] = np.append(dict_users[user_i], idxs_labels[0, img_i])
                label_class[user_i][label_i] += 1
                break

    return dict_users, label_split


def har_iid(dataset, num_users):
    """
    Sample I.I.D. client data from HAR dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """

    print(len(dataset))
    num_items = int(len(dataset) / num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def har_noniid(dataset, num_users=5, label_split=None, default=True, class_on_device=2):
    # labels = np.array(dataset.targets)
    labels = [int(i) for _, i in dataset]
    labels = np.array(labels)

    if default:

        if class_on_device == 2 and num_users == 5:
            label_split = {0: [0, 1, 5], 1: [2, 3], 2: [0, 4, 5], 3: [1, 3], 4: [2, 4]}
        else:
            exit('no default')
    else:
        exit('change the default setting for other conditions')

    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}
    all_imgs = len(labels)
    idxs = np.arange(all_imgs)
    idxs_labels = np.vstack((idxs, labels))

    mask = np.unique(labels)
    num_per_label = {}
    for v in mask:
        num_per_label[v] = np.sum(labels == v)
    tmp = []
    for i in label_split.keys():
        tmp += label_split[i]
    tmp = np.array(tmp)
    mask = np.unique(tmp)
    num_per_class = {}
    for v in mask:
        num_per_class[v] = np.sum(tmp == v)

    label_class = {}

    for i in range(num_users):
        label_class[i] = {}
        for j in label_split[i]:
            label_class[i][j] = 0

    for img_i in range(all_imgs):
        for user_i in range(num_users):
            label_i = idxs_labels[1, img_i]
            total_num_label_i = num_per_label[label_i]
            label_used = num_per_class[label_i]
            if label_i in label_split[user_i] and label_class[user_i][label_i] <= total_num_label_i / label_used:
                dict_users[user_i] = np.append(dict_users[user_i], idxs_labels[0, img_i])
                label_class[user_i][label_i] += 1
                break

    return dict_users, label_split

