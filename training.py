# contains main training loop for all optimizer tests

import pickle
import math
from os import path

import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torch import nn
from torchvision import models
import torch.utils.tensorboard as tb

from get_data import get_mnist_dl, get_cifar10_dl
from models import Mnist_Logistic, save_model
from vis import plot_metrics
from optimizers import (
    SGD_main,
    SGD_momentum,
    RMS_prop,
    Adam,
    Adagrad,
    Adadelta,
    AdamW,
    SGDW,
)

def get_optimizer_class(opt_str, use_pytorch_opt=False):
    """returns the optimizer class given the string from specs"""

    if not use_pytorch_opt:
        opt_map = {
                'SGD': SGD_main,
                'SGDM': SGD_momentum,
                'RMS': RMS_prop,
                'ADAM': Adam,
                'ADAG': Adagrad,
                'ADAD': Adadelta,
                'ADAMW': AdamW,
                'SGDW': SGDW,
            }
    else:
        print('Using the PyTorch optimizers!')
        opt_map = {
                'SGD': torch.optim.SGD,
                'SGDM': torch.optim.SGD,
                'RMS': torch.optim.RMSprop,
                'ADAM': torch.optim.Adam,
                'ADAG': torch.optim.Adagrad,
                'ADAD': torch.optim.Adadelta,
            }
    if not opt_str in opt_map.keys():
        print(f'Optimizer type {opt_str} is unknown.')
        return None
    return opt_map[opt_str]

def get_lr_sched_func(sched_str):
    """returns the learning rate scheduler function given string from specs"""

    sched_map = {
            'step': get_lr_step,
            'inverse_annealing': get_lr_inverse_annealing,
            'root_inverse_annealing': get_lr_root_inverse_annealing,
        }
    if not sched_str in sched_map.keys():
        print(f'LR schedule type {sched_str} is unknown.')
        return None
    return sched_map[sched_str]    

def get_lr_step(start_lr, curr_epoch, total_epochs):
    # parameters to describe the step schedule
    first_step = 0.6
    second_step = 0.85
    first_divisor = 10
    second_divisor = 5

    # determine learning rate for current epoch
    first_cutoff = int(first_step*total_epochs)
    second_cutoff = int(second_step*total_epochs)
    if curr_epoch < first_cutoff:
        return start_lr
    elif first_cutoff <= curr_epoch < second_cutoff:
        return start_lr / first_divisor
    else:
        return start_lr / second_divisor

def get_lr_inverse_annealing(start_lr, curr_epoch, total_epochs):
    return (1/(curr_epoch + 1.))*start_lr

def get_lr_root_inverse_annealing(start_lr, curr_epoch, total_epochs):
    return (1/math.sqrt(curr_epoch + 1.))*start_lr

def momentum_scheduler(m, curr_epoch, total_epochs):
    return m*((1 - ((curr_epoch + 1)/total_epochs))/((1 - m) + m*(1 - ((curr_epoch + 1)/total_epochs))))

def update(x, y, net, opt, loss_func=nn.CrossEntropyLoss()):
    """runs a single batch and returns the loss"""
  
    x = x.to(net.device)
    y = y.to(net.device) 
    opt.zero_grad()
    y_hat = net(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step() # perform the update
    return loss.item()

def main(specs):
    """main training loop"""

    # initialize the tensorboard loggers
    if 'log_dir' in specs.keys():
        train_logger = tb.SummaryWriter(path.join(specs['log_dir'], 'train'))
        valid_logger = tb.SummaryWriter(path.join(specs['log_dir'], 'valid'))

    # get the model
    if specs['model'] == 'LR':
        model = Mnist_Logistic(specs['num_in'], specs['num_out'])
    elif specs['model'] == 'CNN':
        # get resnet18 and change size of input/output layer
        model = models.resnet18(pretrained=False)
        if specs['dataset'] == 'MNIST':
            model.conv1 = torch.nn.Conv2d(
                    in_channels=1, out_channels=64, kernel_size=(7, 7),
                    stride=(2, 2), padding=(3, 3))
        else:
            model.conv1 = torch.nn.Conv2d(
                    in_channels=3, out_channels=64, kernel_size=(7, 7),
                    stride=(2, 2), padding=(3, 3))
        model.fc = torch.nn.Linear(in_features=512, out_features=specs['num_out'])
        model.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    elif specs['model'] == 'BIGCNN':
        # get resnet50 and change size of input/output layer
        # can also initialize as pretrained for faster training/better performance
        model = models.resnet50(pretrained=False)
        if specs['dataset'] == 'MNIST':
            model.conv1 = torch.nn.Conv2d(
                    in_channels=1, out_channels=64, kernel_size=(7, 7),
                    stride=(2, 2), padding=(3, 3))
        else:
            model.conv1 = torch.nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(3, 3),
                stride=(2, 2), padding=(1, 1))
        model.fc = torch.nn.Linear(in_features=2048, out_features=specs['num_out'])
        model.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    else:
        print(f'Model type {specs["model"]} is unknown.')
        return None
    model = model.to(model.device)

    # get the optimizer
    opt = get_optimizer_class(specs['opt'], specs['use_pytorch_opt'])
    if opt is None:
        return None
    if specs['use_pytorch_opt']:
        opt = opt(model.parameters(), **specs['opt_specs'])
    else:
        opt = opt(model, **specs['opt_specs'])

    # get the loss function
    if specs['loss'] == 'CE':
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        print(f'Loss function type {specs["loss"]} is unknown.')
        return None
    
    # get the learning rate scheduling method
    start_lr = specs['opt_specs']['lr']
    lr_sched_func = None
    if specs['lr_sched_type'] is not None:
        lr_sched_func = get_lr_sched_func(specs['lr_sched_type'])
        if lr_sched_func is None:
            return None

    # get the momentum scheduling function
    if specs['use_pytorch_opt']:
        if 'ADAM' in specs['opt']:
            start_m = next(iter(opt.param_groups))['betas'][0]
        elif 'SGD' in specs['opt']:
            start_m = next(iter(opt.param_groups))['momentum']
        else:
            print(f'Momentum scheduling now supported for {specs["opt"]}.')
            return None
    else:
        start_m = opt.m

    momentum_sched_func = None
    if specs['momentum_sched']:
        momentum_sched_func = momentum_scheduler

    # get the data
    if specs['dataset'] == 'MNIST':
        train_dl, valid_dl = get_mnist_dl(**specs['dataset_specs'])
        test_dl = None
    elif specs['dataset'] == 'CIFAR10':
        train_dl, valid_dl, test_dl = get_cifar10_dl(**specs['dataset_specs'])
    else:
        print(f'Dataset {specs["dataset"]} is unknown.')
        return None        

    # main training loop 
    losses = []
    valid_losses = []
    accs = []
    for e in range(specs['epochs']):
        # save model intermittently throughout training
        if e % 10 == 0 and e > 0:
            save_model(model) # saves the model into google drive

        # determine the next learning rate -- store it in optimizer
        if lr_sched_func is not None:
            next_lr = lr_sched_func(start_lr, e, specs['epochs'])
            if specs['use_pytorch_opt']:
                for p in opt.param_groups:
                    p['lr'] = next_lr
            else:
                opt.lr = next_lr

        # determine the next momentum coefficient -- store in optimizer
        if momentum_sched_func is not None:
            next_m = momentum_sched_func(start_m, e, specs['epochs'])
            if specs['use_pytorch_opt']:
                for p in opt.param_groups:
                    if 'ADAM' in specs['opt']:
                        p['betas'] = (next_m, p['betas'][1]) # only change first moment term
                    else:
                        p['momentum'] = next_m
            else:
                opt.m = next_m

        # go through entire training data loader
        model.train()
        if specs['use_tqdm']:
            loss = []
            for x, y in tqdm(train_dl):
                tmp_loss = update(x, y, model, opt, loss_func)
                loss.append(tmp_loss)
        else:
            loss = []
            for x, y in train_dl:
                tmp_loss = update(x, y, model, opt, loss_func)
                loss.append(tmp_loss)
        mean_loss = sum(loss)/len(loss)
        if train_logger is not None:
            train_logger.add_scalar('train_loss', mean_loss, global_step=e)
        print(f'\nEpoch {e} loss: {mean_loss:.4f}')
        losses.append(mean_loss)

        # test validation performance
        with torch.no_grad():
            model.eval()
            total_valid_loss = []
            total_correct = 0.
            total_ex = 0.
            for x, y in valid_dl:
                x = x.to(model.device)
                y = y.to(model.device)
                preds = model(x)
                loss = loss_func(preds, y)
                total_valid_loss.append(loss.item())
                preds = preds.argmax(dim=1)
                num_correct = int(torch.sum((preds == y).float()))
                total_correct += num_correct
                total_ex += y.shape[0]
            valid_loss = sum(total_valid_loss)/len(total_valid_loss)
            valid_acc = total_correct/total_ex
            if valid_logger is not None:
                valid_logger.add_scalar('valid_acc', valid_acc, global_step=e)
                valid_logger.add_scalar('valid_loss', valid_loss, global_step=e)
            valid_losses.append(valid_loss)
            accs.append(valid_acc)
            print(f'\nEpoch {e} Valid Acc.: {valid_acc:.4f}')

    # optionally run evaluation test data
    if test_dl is not None:
        model.eval()
        with torch.no_grad():
            total_correct = 0.
            total_ex = 0.
            for x, y in test_dl:
                x = x.to(model.device)
                y = y.to(model.device)
                preds = model(x)
                preds = preds.argmax(dim=1)
                num_correct = int(torch.sum((preds == y).float()))
                total_correct += num_correct
                total_ex += y.shape[0]
            test_acc = total_correct/total_ex
            print(f'\nTest Accuracy: {valid_acc:.4f}')

    # persist results into an aggregate file to keep track of them
    if specs['persist_results']:
        curr_results = {
                'loss': losses,
                'valid_loss': valid_losses,
                'valid_acc': accs
            }
        persist_results(specs['exp_name'], **curr_results)

    # print a visualization of the results
    if specs['display_results']:
        print(f'Displaying Training Results for {specs["exp_name"]}')
        all_epochs = [x for x in range(len(losses))]
        
        # plot the loss
        plt.title('Training Loss')
        plt.plot(all_epochs, losses)

        # plot the validation accuracy
        plt.title('Validation Accuracy')
        plt.plot(all_epochs, accs)

        # plot the validation loss
        plt.title('Validation Loss')
        plt.plot(all_epochs, valid_losses)

    return losses, accs, valid_losses

def persist_results(exp_name, **kwargs):
    path = '../drive/My Drive/optimizers/exp_results.pckl'
    if os.path.exists(path):
        with open(path, 'rb') as f:
            results_dict = pickle.load(f)
    else:
        results_dict = {}
    results_dict[exp_name] = kwargs
    all_exp_names = results_dict.keys()
    with open(path, 'wb') as f:
        pickle.dump(results_dict, f)
    print(f'Experiment successfully saved.')
    print(f'Currently saved experiments: {all_exp_names}')


if __name__=='__main__':
    # global training parameters
    training_specs = {
            'exp_name': 'test',
            'model': 'BIGCNN',
            'opt': 'SGDM',
            'loss': 'CE',
            #'num_in': 784,
            'num_out': 10,
            'epochs': 200,
            'use_tqdm': True,
            'use_pytorch_opt': True,
            'lr_sched_type': 'step',
            'momentum_sched': True,
            'opt_specs': {
                    'lr': .1,
                    'weight_decay': 1e-4,
                    'momentum': 0.9,
                    #'momentum': 0.,
                    #'m': .9,
                    #'b': .999,
                    #'e': 1e-8,
                },
            'dataset': 'CIFAR10',
            'dataset_specs': {
                    #'flat': False,
                    'bs': 128,
            }
            'display_results': True,
            'persist_results': True,
        }
    main(training_specs)
