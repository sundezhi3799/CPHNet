#!/usr/bin/env python3

""" anoxic score
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import pandas as pd
from conf import settings
from custom_utils import get_network, get_test_dataloader
from custom_train import get_hole_number_prefix_set, load_multi_images


def load_data(directory, size):
    data_list = []
    con_prefixes = get_hole_number_prefix_set(directory)
    for prx_con in list(con_prefixes):
        image_con = load_multi_images(directory, prx_con, size)
        data_list.append([image_con.astype(np.float32), os.path.join(directory, prx_con)])
    return data_list


def process_directory(directory, net, weights, gpu, b, size, class_number, out_dir):
    net.load_state_dict(torch.load(weights))
    net.eval()

    cell_images = load_data(directory, size)
    data_loader = torch.utils.data.DataLoader(dataset=cell_images, batch_size=b, shuffle=False)

    total = 0
    names = []
    outputs = []
    preds = []
    with torch.no_grad():
        for n_iter, (image, name) in enumerate(data_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(data_loader)))

            if gpu:
                image = image.cuda(int(gpu))
            if class_number == 2:
                m = torch.nn.Sigmoid()
            else:
                m = torch.nn.Softmax(dim=1)
            output = m(net(image))
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            outputs += output.tolist()
            names += name
            preds += pred[:, 0].tolist()

    output_array = np.array(outputs)
    # print(output_array)
    results_dct = {'cell': names, 'hypoxia': preds, 'score': output_array[:, 1].tolist()}
    results = pd.DataFrame(results_dct)
    output_file = 'hypoxia_scores_screening_A549_by_both_vgg16.csv'
    results.to_csv(os.path.join(out_dir, output_file), mode='a',
                   header=not os.path.exists(os.path.join(out_dir, output_file)), index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-size', type=int, default=224, help='size of image')
    parser.add_argument('-class_number', type=int, default=2, help='number of classes')
    parser.add_argument('-feature', type=int, default=6, help='number of features')
    args = parser.parse_args()

    net = get_network(args)

    # predict_root = 'data/ours/test_binary/1'  # Change this to the root directory containing multiple subdirectories
    predict_root = 'data/ours/screening/A549_Normal_Control'  # Change this to the root directory containing multiple subdirectories
    out_dir = 'outputs'
    os.makedirs(out_dir, exist_ok=True)

    for subdir in os.listdir(predict_root):
        subdir_path = os.path.join(predict_root, subdir)
        if os.path.isdir(subdir_path):
            if os.listdir(subdir_path):
                print(f"Processing directory: {subdir}")
                process_directory(subdir_path, net, args.weights, args.gpu, args.b, args.size, args.class_number,
                                  out_dir)

    print("Processing complete.")
