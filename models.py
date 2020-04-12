# all model definitions for optimizer tests

# necessary to fix issues with matplotlib on mac
import math
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from os import path
from tqdm import tqdm

from torch import save
import numpy as np
import torch
from torch import nn
from torchvision import models
import matplotlib.pyplot as plt

from optimizers import SGD_momentum, Adam, AdamW
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
            self, start_lr:float, end_lr: float,
            model, opt, loss_func, train_dl, beta=0.95):
        assert start_lr < end_lr
        self.lr_mult = (end_lr/start_lr)**(1/(len(train_dl) - 1))
        self.start_lr = start_lr
        self.end_lr = end_lr
        self.model = model
        self.opt = opt
        self.loss_func = loss_func
        self.train_dl = train_dl
        self.beta = beta
        self.losses = []
        self.log_lrs = []
   
    def lr_find(self):
        # run a little training before lr finder
        # this makes the lr graph easier to read usually
        print('Running initial training...')
        total_batch = [x for x in range(5)]
        for (x, y), _ in tqdm(zip(self.train_dl, total_batch)):
            self.opt.zero_grad()
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            loss.backward()
            self.opt.step()

        # run a batch with all different learning rates over an epoch
        print('Running lr sweep...')
        best_loss = None
        exp_avg_loss = 0.
        lr_curr = self.start_lr
        self.losses = [] # always clear losses/lrs before next test
        self.log_lrs = []
        i = 1
        for x, y in tqdm(self.train_dl):
            # set lr and calculate loss over batch
            self.opt.lr = lr_curr
            self.opt.zero_grad()
            y_hat = self.model(x)
            loss = self.loss_func(y_hat, y)
            loss.backward()
            
            # compute exponential average for the loss
            exp_avg_loss = self.beta * exp_avg_loss + (1 - self.beta) * loss.item()
            debias_loss = exp_avg_loss / (1 - self.beta**i)
            if best_loss is None:
                best_loss = debias_loss
            elif debias_loss > 5 * best_loss:
                break
            elif debias_loss < best_loss:
                best_loss = debias_loss
            self.losses.append(debias_loss)
            self.log_lrs.append(math.log10(lr_curr))

            # update parameters and learning rate
            self.opt.step()
            lr_curr *= self.lr_mult
            i += 1

    def lr_plot(self, title='Learning Rate Finder'):
        assert len(self.losses) > 0 and len(self.losses) == len(self.log_lrs)
        plt.plot(self.log_lrs, self.losses)
        plt.xlabel("Learning Rate (log scale)")
        plt.ylabel(f"Loss, Beta={self.beta:.4f}")
        plt.title(title)
        plt.show()

    def get_metrics(self):
        """useful if you want to run multiple lr finders with different
        settings and plot them together"""

        assert len(self.losses) > 0 and len(self.losses) == len(self.log_lrs)
        return self.log_lrs, self.losses

def save_model(model):
    return save(model.state_dict(), '../drive/My Drive/optimizers/cf10.th')

if __name__=='__main__':
    # get everything needed for lr finder
    colors = ['r', 'g', 'b', 'y', 'k']
    start_lr = 1e-4
    end_lr = .1
    beta = 0.98
    loss_func = torch.nn.CrossEntropyLoss()
    #decays = [.003, .01, .03, .1]
    losses = []
    log_lrs = []
    for wd in decays:
        model = models.resnet50(pretrained=False)
        model.conv1 = torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(3, 3),
                stride=(2, 2), padding=(1, 1))
        model.fc = torch.nn.Linear(in_features=2048, out_features=10)
        model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt = AdamW(model, lr=start_lr, wd=wd)
        train_dl, _, _ = get_cifar10_dl(**{'bs': 128})

        # run the lr finder
        lr_finder = LR_finder(start_lr, end_lr, model, opt, loss_func, train_dl, beta)
        lr_finder.lr_find()
        log_lr, loss = lr_finder.get_metrics()
        losses.append(loss)
        log_lrs.append(log_lr)


    # plot all of the entries in the lr finder together
    plt.xlabel('Learning Rate (log scale)')
    plt.ylabel(f'Loss, Beta={beta:.4f}')
    print(losses)
    print(log_lrs)
    print(decays)
    print(colors)
    for c, loss, log_lr, wd in zip(colors, losses, log_lrs, decays):
        exp_name = f'WD={wd}'
        plt.plot(log_lr, loss, color=c, label=exp_name)
        plt.legend()
    plt.show()
