# helper methods for getting mnist and other kinds of data

import pickle
import gzip
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

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
    data = torch.tensor(data, dtype=torch.float)
    labels = torch.tensor(labels)
    trn_ds = TensorDataset(data, labels)
    trn_dl = DataLoader(trn_ds, batch_size=bs)

    # get the validation data
    valid_data = None
    valid_labels = None
    with open(path + 'data_batch_5', 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        valid_data = data_dict[b'data']
        valid_labels = data_dict[b'labels']
    valid_data = reshape_cifar_data(valid_data)
    valid_data = torch.tensor(valid_data, dtype=torch.float)
    valid_labels = torch.tensor(valid_labels, dtype=torch.float)
    valid_ds = TensorDataset(valid_data, valid_labels)
    valid_dl = DataLoader(valid_ds, batch_size=bs)

    # get the test data
    test_data = None
    test_labels = None
    with open(path + 'test_batch', 'rb') as fo:
        data_dict = pickle.load(fo, encoding='bytes')
        test_data = data_dict[b'data']
        test_labels = data_dict[b'labels']
    test_data = reshape_cifar_data(test_data)
    test_data = torch.tensor(test_data, dtype=torch.float)
    test_labels = torch.tensor(test_labels)
    test_ds = TensorDataset(test_data, test_labels)
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
