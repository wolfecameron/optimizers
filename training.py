# contains main training loop for all optimizer tests

import torch
from torch import nn

from get_data import get_mnist_dl
from models import (
    Mnist_Logistic,
)
from optimizers import (
    SGD_weight_decay,
    SGD_momentum,
    RMS_prop,    
    Adam,
    Adagrad,
    Adadelta,
)
from vis import (
    plot_metrics,
)

def get_optimizer_class(opt_str):
    """returns the optimizer class given the string from specs"""

    opt_map = {
            'SGDWD': SGD_weight_decay,
            'SGDM': SGD_momentum,
            'RMS': RMS_prop,
            'ADAM': Adam,
            'ADAG': Adagrad,
            'ADAD': Adadelta,
        }
    if not opt_str in opt_map.keys():
        print(f'Optimizer type {opt_str} is unknown.')
        return None
    return opt_map[opt_str]

def update(x, y, net, opt, loss_func=nn.CrossEntropyLoss()):
    """runs a single batch and returns the loss"""
    
    opt.zero_grad()
    y_hat = net(x)
    wdl = opt.wd_loss() # weight decay loss
    loss = loss_func(y_hat, y) + wdl # combine the weight decay and normal loss
    loss.backward()
    opt.step() # perform the update
    return loss.item()

def main(specs):
    """main training loop"""

    # get the model
    if specs['model'] == 'LR':
        model = Mnist_Logistic(specs['num_in'], specs['num_out'])
    else:
        print(f'Model type {specs["model"]} is unknown.')
        return None

    # get the optimizer
    opt = get_optimizer_class(specs['opt'])
    if opt is None:
        return None
    opt = opt(model, **specs['opt_specs'])

    # get the loss function
    if specs['loss'] == 'CE':
        loss_func = torch.nn.CrossEntropyLoss()
    else:
        print(f'Loss function type {specs["loss"]} is unknown.')
        return None

    # get the data
    if specs['dataset'] == 'MNIST':
        train_dl, valid_dl = get_mnist_dl(specs['bs'])
    else:
        print(f'Dataset {specs["dataset"]} is unknown.')
        return None        
   
    losses = []
    accs = []
    for e in range(specs['epochs']):
        # go through entire training data loader
        loss = [update(x, y, model, opt) for x, y in train_dl]
        mean_loss = sum(loss)/len(loss)
        print(f'Epoch {e} loss: {mean_loss:.4f}')
        losses.append(mean_loss)

        # test validation performance
        with torch.no_grad():
            model.eval()
            total_correct = 0.
            total_ex = 0.
            for x, y in valid_dl:
                preds = model(x)
                preds = preds.argmax(dim=1)
                num_correct = torch.sum((preds == y).float())
                total_correct += num_correct
                total_ex += y.shape[0]
            valid_acc = total_correct/total_ex
            accs.append(valid_acc)
            print(f'Epoch {e} Valid Acc.: {valid_acc:.4f}')
    return losses, accs

# different methods for experiments with optimizers

def sgdwd_diff_lr(training_specs):
    """experiment for SGD with weight decay with a bunch of different learning rates"""

    loss_results = {}
    acc_results = {}
    lrs = [1e-4, 1e-3, 1e-2, .1, 1.0, 10.0]
    for lr in lrs:
        exp_name = f'SGDWD: LR={lr:.4f}'
        training_specs['opt_specs']['lr'] = lr
        losses, accs = main(training_specs)
        loss_results[exp_name] = losses
        acc_results[exp_name] = accs
    plot_metrics(loss_results, title='Training Losses, SGD')
    plot_metrics(acc_results, ylabel='Accuracy', title='Validation Accuracy, SGD')

def sgdm_diff_lr(training_specs):
    loss_results = {}
    acc_results = {}
    lrs = [1e-4, 1e-3, 1e-2, .1, 1.0, 10.0]
    training_specs['opt'] = 'SGDM'
    training_specs['opt_specs']['m'] = 0.9 # very typical choice
    for lr in lrs:
        exp_name = f'SGD w/ Momentum: LR={lr:.4f}'
        training_specs['opt_specs']['lr'] = lr
        losses, accs = main(training_specs)
        loss_results[exp_name] = losses
        acc_results[exp_name] = accs
    plot_metrics(loss_results, title='Training Losses, SGD w/ Momentum')
    plot_metrics(acc_results, ylabel='Accuracy', title='Validation Accuracy, SGD w/ Momentum')

def rmsprop_diff_lr(training_specs):
    loss_results = {}
    acc_results = {}
    lrs = [1e-4, 1e-3, 1e-2, .1, 1.0, 10.0]
    training_specs['opt'] = 'RMS'
    training_specs['opt_specs']['b'] = 0.9 # very typical choice
    for lr in lrs:
        exp_name = f'RMS PROP: LR={lr:.4f}'
        training_specs['opt_specs']['lr'] = lr
        losses, accs = main(training_specs)
        loss_results[exp_name] = losses
        acc_results[exp_name] = accs
    plot_metrics(loss_results, title='Training Losses, RMSProp')
    plot_metrics(acc_results, ylabel='Accuracy', title='Validation Accuracy, RMSProp')

def adam_diff_lr(training_specs):
    loss_results = {}
    acc_results = {}
    lrs = [1e-4, 1e-3, 1e-2, .1, 1.0, 10.0]
    training_specs['opt'] = 'ADAM'
    training_specs['opt_specs']['m'] = 0.9
    training_specs['opt_specs']['b'] = 0.999
    for lr in lrs:
        exp_name = f'ADAM: LR={lr:.4f}'
        training_specs['opt_specs']['lr'] = lr
        losses, accs = main(training_specs)
        loss_results[exp_name] = losses
        acc_results[exp_name] = accs
    plot_metrics(loss_results, title='Training Losses, Adam')
    plot_metrics(acc_results, ylabel='Accuracy', title='Validation Accuracy, Adam')

def adag_diff_lr(training_specs): 
    loss_results = {}
    acc_results = {}
    lrs = [1e-4, 1e-3, 1e-2, .1, 1.0, 10.0]
    training_specs['opt'] = 'ADAG'
    for lr in lrs:
        exp_name = f'ADAGRAD: LR={lr:.4f}'
        training_specs['opt_specs']['lr'] = lr
        losses, accs = main(training_specs)
        loss_results[exp_name] = losses
        acc_results[exp_name] = accs
    plot_metrics(loss_results, title='Training Losses, Adagrad')
    plot_metrics(acc_results, ylabel='Accuracy', title='Validation Accuracy, Adagrad')

if __name__=='__main__':
    # global training parameters
    training_specs = {
            'model': 'LR',
            'opt': 'ADAD',
            'loss': 'CE',
            'num_in': 784,
            'num_out': 10,
            'epochs': 45,
            'opt_specs': {
                    #'lr': 3e-2,
                    'wd': 1e-4,
                    'm': .9,
                    #'b': .999,
                    'e': 1e-8,
                },
            'dataset': 'MNIST',
            'bs': 128,
        }
    losses, acc = main(training_specs)
    plot_metrics({'ADADELTA': losses}, title='Training Losses, Adadelta')
    plot_metrics({'ADADELTA': acc}, ylabel='Accuracy', title='Validation Accuracy, Adadelta')
