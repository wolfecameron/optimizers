# contains main training loop for all optimizer tests

import torch

from get_data import get_mnist_dl
from models import (
    Mnist_Logistic,
)
from optimizers import (
    SGD_weight_decay,
    SGD_momentum,
    RMS_prop,    
    Adam,
)

# global training parameters
training_specs = {
        'model': 'LR',
        'opt': 'SGDWD',
        'loss': 'CE',
        'num_in': 784,
        'num_out': 10,
        'epochs': 15,
        'opt_specs': {
                'lr': 3e-3,
                'wd': 1e-5,
                'm': .9,
                'b': .999,
                'e': 1e-8,
            }
        'dataset': 'MNIST',
        'bs': 128,
    }



def get_optimizer_class(opt_str):
    """returns the optimizer class given the string from specs"""

    opt_map = {
            'SGDWD': SGD_weight_decay,
            'SGDM': SGD_momentum,
            'RMS': RMS_prop,
            'ADAM': Adam,
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
        trn_dl, valid_dl = get_mnist_dl(specs['bs'])
    else:
        print(f'Dataset {specs["dataset"]} is unknown.')
        return None        
   
    losses = [] 
    for e in range(epochs):
        # go through entire training data loader
        loss = [update(x, y, log_net, opt) for x, y in train_dl]
        losses.append(sum(loss)/len(loss))

    # test validation performance
    with torch.no_grad:
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
        print(f'Final Validation Acc.: {valid_acc:.4f}')
    return losses, valid_acc
