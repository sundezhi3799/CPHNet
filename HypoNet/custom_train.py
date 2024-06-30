# train.py
#!/usr/bin/env	python3

""" train network using pytorch

"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
import argparse
import time
from datetime import datetime
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from conf import settings
from custom_utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

class arg():
    def __init__(self, net, gpu, b, warm, lr, resume, size, feature=4, class_number=2):
        self.net = net
        self.gpu = gpu
        self.b = b
        self.warm = warm
        self.lr = lr
        self.resume = resume
        self.size = size
        self.feature = feature
        self.class_number = class_number


def train(epoch,training_loader):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(training_loader):
        # labels=labels.
        if args.gpu:
            labels = labels.cuda(int(args.gpu))
            images = images.cuda(int(args.gpu))

        optimizer.zero_grad()
        if args.class_number==2:
            m=nn.Sigmoid()
        else:
            m=nn.Softmax(dim=1)
        outputs = m(net(images))
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(training_loader.dataset)
        ),end='\r')

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch, test_loader, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    all_labels = []
    all_preds = []

    for (images, labels) in test_loader:

        if args.gpu:
            images = images.cuda(int(args.gpu))
            labels = labels.cuda(int(args.gpu))
        if args.class_number==2:
            m=nn.Sigmoid()
        else:
            m=nn.Softmax(dim=1)
        outputs = m(net(images))
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        _, label_indices=labels.max(1)
        correct += preds.eq(label_indices).sum()

        all_labels.extend(label_indices.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    finish = time.time()

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)

    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1 Score: {:.4f}, AUC: {:.4f}, Time consumed: {:.2f}s'.format(
        epoch,
        test_loss / len(test_loader.dataset),
        correct.float() / len(test_loader.dataset),
        precision,
        recall,
        f1,
        auc,
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(test_loader.dataset), epoch)
        writer.add_scalar('Test/Precision', precision, epoch)
        writer.add_scalar('Test/Recall', recall, epoch)
        writer.add_scalar('Test/F1 Score', f1, epoch)
        writer.add_scalar('Test/AUC', auc, epoch)

    return correct.float() / len(test_loader.dataset)

def load_multi_images(dir,hole_number,size):
    path_dna=os.path.join(dir, hole_number + '_dna.png')
    path_pm=os.path.join(dir, hole_number + '_pm.png')
    path_er=os.path.join(dir, hole_number + '_er.png')
    path_mit=os.path.join(dir, hole_number + '_mit.png')
    path_ag=os.path.join(dir, hole_number + '_ag.png')
    path_rna=os.path.join(dir, hole_number + '_rna.png')
    image_dna=(cv2.imread(path_dna,cv2.IMREAD_GRAYSCALE)/255.0-0.5)*2
    image_pm=(cv2.imread(path_pm,cv2.IMREAD_GRAYSCALE)/255.0-0.5)*2
    image_er=(cv2.imread(path_er,cv2.IMREAD_GRAYSCALE)/255.0-0.5)*2
    image_mit=(cv2.imread(path_mit,cv2.IMREAD_GRAYSCALE)/255.0-0.5)*2
    image_ag=(cv2.imread(path_ag,cv2.IMREAD_GRAYSCALE)/255.0-0.5)*2
    image_rna=(cv2.imread(path_rna,cv2.IMREAD_GRAYSCALE)/255.0-0.5)*2
    image=cv2.merge([image_dna,image_er,image_ag,image_mit,image_pm,image_rna])
    image = cv2.resize(image, (size, size))
    image=image.reshape(6,image.shape[0],image.shape[1])
    return image

def get_hole_number_prefix_set(directory):
    filenames = os.listdir(directory)
    name_set = set()
    for filename in filenames:
        name_parts = filename.split('_')
        if len(name_parts) > 1:
            name_set.add(name_parts[0]+'_'+name_parts[1]+'_'+name_parts[2])
    return name_set

def load_data(directory,size):
    data_list=[]
    classes=os.listdir(directory)
    for class_dir in classes:
        root=os.path.join(directory,class_dir)
        for sub_dir in os.listdir(root):
            sub_dir_path=os.path.join(root,sub_dir)
            con_prefixes=get_hole_number_prefix_set(sub_dir_path)
            for prx_con in list(con_prefixes):
                image_con=load_multi_images(sub_dir_path,prx_con,size)
                y = np.array([0.] * len(classes))
                y[int(class_dir)]=1.
                data_list.append([image_con.astype(np.float32),y.astype(np.float32)])
    return data_list

def split_train_test_set(full_dataset,test_prop=0.2):
    train_size = int((1-test_prop) * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train, test = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    train_dataset=[full_dataset[i] for i in train.indices]
    test_dataset=[full_dataset[i] for i in test.indices]
    return train_dataset,test_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-size', type=int, default=224, help='size of image')
    parser.add_argument('-class_number', type=int, default=2, help='number of classes')
    parser.add_argument('-feature', type=int, default=6, help='number of features')
    args = parser.parse_args()

    net = get_network(args)

    cell_images=load_data('data/ours/train_binary_224',args.size)

    train_set,test_set=split_train_test_set(cell_images,0.2)

    training_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=args.b, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=args.b, shuffle=True)
    if args.class_number>2:
        loss_function = nn.CrossEntropyLoss()
    else:
        loss_function=nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.5) #learning rate decay
    iter_per_epoch = len(train_set)/args.b
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(args.b, args.feature, args.size, args.size)
    if args.gpu:
        input_tensor = input_tensor.cuda(int(args.gpu))
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step()
        train(epoch,training_loader=training_loader)
        acc = eval_training(epoch,testing_loader)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
