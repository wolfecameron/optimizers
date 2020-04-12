# helper methods for getting mnist and other kinds of data

import pickle
import gzip
from pathlib import Path

import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision
from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Dataset,
)

def get_mnist_data(flat:bool=True):
    path = Path('./data_files/')
    with gzip.open(path/'mnist.pkl.gz', 'rb') as f:
         ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    if not flat:
        x_train = x_train.reshape((-1, 1, 28, 28))
        x_valid = x_valid.reshape((-1, 1, 28, 28))
    return (x_train, y_train, x_valid, y_valid)

def get_mnist_dl(flat=True, bs=128):
    x_train, y_train, x_valid, y_valid = get_mnist_data(flat)
    trn_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs)
    return trn_dl, valid_dl

def get_cifar10_dl(path='./data_files/cifar-10-batches-py/', bs=128):
    """method found at https://www.cs.toronto.edu/~kriz/cifar.html"""

    # get the training data
    file_names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4']
    labels = []
    data = []
    for fn in file_names:
        with open(path + fn, 'rb') as fo:
            data_dict = pickle.load(fo, encoding='bytes')
            labels.append(data_dict[b'labels'])
            data.append(data_dict[b'data']) 
    labels = np.concatenate(labels, axis=0)
    data = np.concatenate(data, axis=0)
    data = reshape_cifar_data(data)
    data = torch.tensor(data, dtype=torch.float)/255. # normalize px between 0 and 1
    labels = torch.tensor(labels)
    trn_ds = CifarDataset(data, labels, True)
    trn_dl = DataLoader(trn_ds, batch_size=bs)

    # get the validation data
    valid_data = None
    valid_labels = None
    with open(path + 'data_batch_5', 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        valid_data = data_dict[b'data']
        valid_labels = data_dict[b'labels']
    valid_data = reshape_cifar_data(valid_data)
    valid_data = torch.tensor(valid_data, dtype=torch.float)/255.
    valid_labels = torch.tensor(valid_labels, dtype=torch.float)
    valid_ds = CifarDataset(valid_data, valid_labels, False)
    valid_dl = DataLoader(valid_ds, batch_size=bs)

    # get the test data
    test_data = None
    test_labels = None
    with open(path + 'test_batch', 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        test_data = data_dict[b'data']
        test_labels = data_dict[b'labels']
    test_data = reshape_cifar_data(test_data)
    test_data = torch.tensor(test_data, dtype=torch.float)/255.
    test_labels = torch.tensor(test_labels)
    test_ds = CifarDataset(test_data, test_labels, False)
    test_dl = DataLoader(test_ds, batch_size=bs)
    return trn_dl, valid_dl, test_dl

def reshape_cifar_data(img, to_view:bool=False):
    """reshapes cifar10 batch or single image into the proper shape --
    img may either be a single img or an entire batch"""

    if len(img.shape) == 2:
        img = img.reshape((img.shape[0], 3, 32, 32))
        if to_view:
            img = img.transpose(0, 2, 3, 1)
    else:
        img = img.reshape(3, 32, 32)
        if to_view:
            img = img.transpose(1, 2, 0)
    return img

class CifarDataset(Dataset):
    def __init__(self, tensor:torch.tensor, labels:torch.tensor, use_trans:bool=True):
        super().__init__()
        self.tensor = tensor
        self.labels = labels
        
        # mean/std stats are borrowed from the fast.ai cifar10 notebooks
        self.stats = (
                torch.tensor([0.4914, 0.48216, 0.44653]),
                torch.tensor([0.24703, 0.24349, 0.26159]))

        # during training, use the full data augmentation
        # for now, don't use color jitter because not used in papers I'm trying to match
        if use_trans:
            self.trans = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
                    torchvision.transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.stats[0], self.stats[1])])

        # during testing, only normalize the images
        else:
            self.trans = torchvision.transforms.Compose([
                    torchvision.transforms.ToPILImage(),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(self.stats[0], self.stats[1])])

    def __len__(self):
        return self.tensor.shape[0]

    def __getitem__(self, index):
        data = self.tensor[index, :]
        label = self.labels[index]
        return self.trans(data), label