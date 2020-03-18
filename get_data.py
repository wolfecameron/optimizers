# helper methods for getting mnist and other kinds of data

import pickle
import gzip
from pathlib import Path

import torch
from torch.utils.data import TensorDataset, DataLoader

def get_mnist_data():
    path = Path('./data_files/')
    with gzip.open(path/'mnist.pkl.gz', 'rb') as f:
         ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')
    x_train, y_train, x_valid, y_valid = map(torch.tensor, (x_train, y_train, x_valid, y_valid))
    return (x_train, y_train, x_valid, y_valid)

def get_mnist_dl(bs=128):
    x_train, y_train, x_valid, y_valid = get_mnist_data()
    trn_ds = TensorDataset(x_train, y_train)
    valid_ds = TensorDataset(x_valid, y_valid)
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=bs)
    return trn_dl, valid_dl
