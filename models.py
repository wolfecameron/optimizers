# all model definitions for optimizer tests

import torch
from torch import nn

class Mnist_Logistic(nn.Module):
    def __init__(self, num_in=784, num_out=10):
        super().__init__()
        self.lin = nn.Linear(num_in, num_out)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self, xb):
        return self.lin(xb)
