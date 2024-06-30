#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

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
from custom_train import split_train_test_set,get_hole_number_prefix_set,load_multi_images

def load_data(directory,size):
    data_list=[]
    prefixes=get_hole_number_prefix_set(directory)
    for prx in prefixes:
        image=load_multi_images(directory,prx,size)
        data_list.append([image.astype(np.float32),os.path.join(directory,prx)])
    return data_list

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-size', type=int, default=224, help='size of image')
    args = parser.parse_args()

    net = get_network(args)
    predict_dir='data/ours/test2'
    cell_images = load_data(predict_dir, args.size)
    # training_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.b, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=cell_images, batch_size=args.b, shuffle=False)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    names=[]
    outputs=[]
    preds=[]
    with torch.no_grad():
        for n_iter, (image, name) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')
            m=torch.nn.Sigmoid()
            output = m(net(image))
            _, pred = output.topk(1, 1, largest=True, sorted=True)
            outputs+=output.tolist()
            names+=name
            preds+=pred[:,0].tolist()
    output_array=np.array(outputs)
    results_dct={'cell':names,'pred':preds,0:output_array[:,0].tolist(),1:output_array[:,1].tolist()}
    results=pd.DataFrame(results_dct)
    results.to_csv(os.path.join(predict_dir,'preds.csv'),index=False)
    pass