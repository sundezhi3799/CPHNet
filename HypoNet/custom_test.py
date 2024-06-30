#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

"""

import argparse

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from custom_utils import get_network, get_test_dataloader
from custom_train import load_data,split_train_test_set
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    parser.add_argument('-size', type=int, default=224, help='size of image')
    parser.add_argument('-class_number', type=int, default=2, help='number of classes')
    args = parser.parse_args()


    args=arg(net='resnet50',weights='checkpoint/resnet50/Friday_10_November_2023_14h_38m_57s/resnet50-10-regular.pth',gpu='0',b=16,size=224,class_number=2)
    net = get_network(args)

    test_set = load_data('data/ours/test_binary', args.size)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.b, shuffle=True)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                # print('GPU INFO.....')
                # print(torch.cuda.memory_summary(), end='')
            if args.class_number == 2:
                m = nn.Sigmoid()
            else:
                m = nn.Softmax(dim=1)
            output = m(net(image))
            _, preds = output.max(1)
            _, label_indices = label.max(1)
            correct += preds.eq(label_indices).sum()




        if args.gpu:
            print('GPU INFO.....')
            print(torch.cuda.memory_summary(), end='')

        print()
        print("Test acc: ",  correct / len(test_loader.dataset))
        print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))