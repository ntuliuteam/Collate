#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7.7

import sys
from tensorflow.python.framework import graph_util
import tensorflow as tf
import numpy as np
import time
import argparse


def create_graph(model_path):
    with tf.io.gfile.GFile(model_path, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        tensor_name_list = [tensor.name for tensor in tf.compat.v1.get_default_graph().as_graph_def().node]
        input_tensor = tensor_name_list[0] + ":0"
        output_tensor = tensor_name_list[-1] + ":0"

        return input_tensor, output_tensor


def run_main(model_path, input_size):
    detection_graph = tf.Graph()
    tf.reset_default_graph()
    with detection_graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        config.gpu_options.allow_growth = True
        input_tensor, output_tensor = create_graph(model_path)

        # run_metadata = tf.RunMetadata()

        with tf.Session(graph=detection_graph, config=config) as sess:
            image_tensor = detection_graph.get_tensor_by_name(output_tensor)

            if len(input_size) < 3:
                image = np.random.rand(input_size[0], input_size[1])
                images = [image]
            else:
                image = np.random.rand(input_size[0], input_size[1], input_size[2])
                images = [image]

            for i in range(100):
                sess.run(image_tensor, {input_tensor: images})  # reduce the influence of first few times

            start = time.time()

            for i in range(100):
                sess.run(image_tensor, {input_tensor: images})
            print((time.time() - start) * 10, 'ms')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="The path to a frozen model file.", required=True)
    parser.add_argument('--dataset', type=str, default='cifar100', help='training dataset')
    args = parser.parse_args()

    input_size = {
        'cifar10': [224, 224, 3],
        'cifar100': [32, 32, 3],
        'mnist': [28, 28, 1],
        'Har': [128, 9]
    }

    model_path = args.model_path
    run_main(model_path, input_size[args.dataset])
