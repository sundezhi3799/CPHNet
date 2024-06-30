#!/usr/bin/env python3

""" anoxic score """

import argparse
import os
import numpy as np
import pandas as pd
import torch
from custom_utils import get_network
from custom_train import get_hole_number_prefix_set, load_multi_images

# Function to load data and extract features
def load_data_with_features(directory, output_directory, size, net, gpu):
    classes = os.listdir(directory)

    # Set the network in evaluation mode
    net.eval()

    with torch.no_grad():
        for class_dir in classes:
            root = os.path.join(directory, class_dir)
            for sub_dir in os.listdir(root):
                sub_dir_path = os.path.join(root, sub_dir)
                con_prefixes = get_hole_number_prefix_set(sub_dir_path)

                features_list = []  # List to store features for the current sub_dir

                for prx_con in list(con_prefixes):
                    image_con = load_multi_images(sub_dir_path, prx_con, size)
                    image_tensor = torch.from_numpy(image_con.astype(np.float32)).unsqueeze(0)

                    if gpu:
                        image_tensor = image_tensor.cuda()

                    # Get the features from the last layer of the network
                    features = net.toplayer(image_tensor).cpu().numpy()
                    features_list.append([features.flatten(), os.path.join(sub_dir, prx_con)])

                # Save features to a CSV file for the current sub_dir
                features, names = zip(*features_list)
                features_df = pd.DataFrame(features, columns=[f'feature_{i}' for i in range(len(features[0]))])
                features_df['cell'] = names
                features_csv_path = os.path.join(output_directory, f'{sub_dir}_resnet_features.csv')
                features_df.to_csv(features_csv_path, index=False)

if __name__ == '__main__':
    class arg():
        def __init__(self, net, weights, gpu, b, size, class_number):
            self.net = net
            self.weights = weights
            self.gpu = gpu
            self.b = b
            self.size = size
            self.class_number = class_number

    args = parser.parse_args()

    net = get_network(args)
    net.load_state_dict(torch.load(args.weights))

    predict_dir = 'data/ours/test_binary'
    features_out_dir = 'features/1106_A549_1'
    os.makedirs(features_out_dir, exist_ok=True)

    # Load data and extract features
    load_data_with_features(predict_dir,features_out_dir, args.size, net, args.gpu)
    # Read all individual CSV files and merge into a single dataframe
    all_features_df = pd.DataFrame()  # Create an empty dataframe to store all features

    for sub_dir_csv in os.listdir(features_out_dir):
        if sub_dir_csv.endswith('_resnet_features.csv'):
            sub_dir_csv_path = os.path.join(features_out_dir, sub_dir_csv)
            sub_dir_features_df = pd.read_csv(sub_dir_csv_path)
            all_features_df = pd.concat([all_features_df, sub_dir_features_df], ignore_index=True)

    # Save the merged dataframe to a CSV file
    all_features_df.to_csv(os.path.join(features_out_dir, '1106_resnet_features.csv'), index=False)


