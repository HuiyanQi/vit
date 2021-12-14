import os
import sys
import json
import pickle
import random

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

import argparse

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.025)
    parser.add_argument('--lrf', type=float, default=0.01)

    # http://download.tensorflow.org/example_images/flower_photos.tgz
    #parser.add_argument('--data_path', type=str,
    #                    default="./data/flower_photos/")
    parser.add_argument('--root', type=str,
                       default="/workspace/qhy/CUB/CUB_200_2011/dataset/")
    parser.add_argument('--model_name', default='', help='create model name')

    parser.add_argument('--weights', type=str, default='./vit_base_patch16_224_in21k.pth',
                        help='initial weights path')
    parser.add_argument('--model_weights_dir', type=str, default='./weights') 
    parser.add_argument('--freeze_layers', type=bool, default=False)
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    args = parser.parse_args()

    return args
    


args = parse()
def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    flower_class.sort()
    class_indices = dict((k, v) for v, k in enumerate(flower_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  
    train_images_label = []
    val_images_path = []  
    val_images_label = []  
    every_class_num = []
    supported = [".jpg", ".JPG", ".png", ".PNG"]
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                                  if os.path.splitext(i)[-1] in supported]
        image_class = class_indices[cla]
        every_class_num.append(len(images))
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        plt.bar(range(len(flower_class)), every_class_num, align='center')
        plt.xticks(range(len(flower_class)), flower_class)
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        plt.xlabel('image class')
        plt.ylabel('number of images')
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  
            plt.yticks([])  
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def smooth(targets: torch.Tensor, classes: int, smoothing):
    targets = targets.resize_(args.batch_size, 1)
    one_hot = torch.zeros(args.batch_size, args.num_classes).scatter_(1, targets, 1)

    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((one_hot.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape)    
        true_dist.fill_(smoothing / (classes - 1))
        _, index = torch.max(one_hot, 1)
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)
    return true_dist

def celoss(outputs,targets):
    logsoftmax_func = nn.LogSoftmax(dim=1)
    outputs = logsoftmax_func(outputs)
    loss = targets.mul(outputs)
    loss = -torch.sum(loss, dim = 1).mean()
    return loss

def train_one_epoch(model, optimizer, data_loader, epoch,scheduler):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    total_train_acc = 0
    total_train_loss = 0
    optimizer.zero_grad()

    data_loader = tqdm(data_loader)

    for step, data in enumerate(data_loader):
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        pred = model(images)

        acc = accuracy(pred, labels)[0]
        loss = loss_function(pred,labels)
        loss.backward()
        total_train_acc += acc.item()
        total_train_loss += loss.item()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.2f}%".format(epoch,
                                                                               total_train_loss / (step + 1),
                                                                               total_train_acc / (step + 1))

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_train_loss / (step + 1), total_train_acc / (step + 1)


#@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    total_test_acc = 0
    total_test_loss = 0

    data_loader= tqdm(data_loader)
    with torch.no_grad():
        for step, data in enumerate(data_loader):
            images, labels = data
            images, labels = images.cuda(), labels.cuda()
            pred = model(images)

            acc = accuracy(pred, labels)[0]
            loss = loss_function(pred, labels)
            total_test_acc += acc.item()
            total_test_loss += loss.item()

            data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.2f}%".format(epoch,
                                                                                   total_test_loss / (step + 1),
                                                                                   total_test_acc / (step + 1))

    return total_test_loss / (step + 1), total_test_acc / (step + 1)


