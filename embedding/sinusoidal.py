import torch
import torch.nn as nn

class Sin(nn.Module):

    def __init__(self):
        super(Sin, self).__init__()
        return

    def forward(self, x):
        return torch.sin(x)
