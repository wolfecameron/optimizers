# all model definitions for optimizer tests

# necessary to fix issues with matplotlib on mac
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm

import numpy as np
import torch
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt

from optimizers import SGD_momentum, Adam
from get_data import get_cifar10_dl

class Mnist_Logistic(nn.Module):
    def __init__(self, num_in=784, num_out=10):
        super().__init__()
        self.lin = nn.Linear(num_in, num_out)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, xb):
        return self.lin(xb)

class LR_finder():
    def __init__(
            self, start_lr:float, end_lr: float, num_it:int,
            model, opt, loss_func, train_dl):
        assert start_lr < end_lr
        self.lr_sched = np.arange(start_lr, end_lr, (end_lr - start_lr)/num_it)
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.train_dl = train_dl
        self.losses = []
   
    def lr_find(self):
        # run a little training before lr finder
        # this makes the lr graph easier to read usually
        print('Running initial training...')
        total_batch = [x for x in range(50)] # only run 100 batch of training
        for (x, y), _ in tqdm(zip(self.train_dl, total_batch)):
            self.opt.zero_grad()
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            loss.backward()
            self.opt.step()

        # run a mini batch with each learning rate, keep track of loss
        self.losses = [] # always clear losses before next test
        for lr_curr in tqdm(self.lr_sched):
            self.opt.lr = lr_curr
            self.opt.zero_grad()
            x, y = next(iter(self.train_dl)) # always run on same batch
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            loss.backward()
            self.losses.append(loss.item())
            self.opt.step()
  
    def lr_plot(self, title='Learning Rate Finder'):
        assert len(self.losses) > 0
        plt.plot(self.lr_sched, self.losses)
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss")
        plt.title(title)
        plt.show()

if __name__=='__main__':
    # get everything needed for lr finder
    model = models.resnet18(pretrained=False)
    model.conv1 = torch.nn.Conv2d(
            in_channels=3, out_channels=64, kernel_size=(7, 7),
            stride=(2, 2), padding=(3, 3))
    model.fc = torch.nn.Linear(in_features=512, out_features=10)
    model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    opt = SGD_momentum(model, lr=3e-3)
    loss_func = torch.nn.CrossEntropyLoss()
    train_dl, _, _ = get_cifar10_dl(**{'bs': 64})

    # run the lr finder
    start_lr = 1e-3
    end_lr = .3
    num_it = 20
    lr_finder = LR_finder(start_lr, end_lr, num_it, model, opt, loss_func, train_dl)
    lr_finder.lr_find()
    lr_finder.lr_plot('Adam LR Finder')
