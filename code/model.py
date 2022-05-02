import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Creates Model class
class Model(nn.Module):
    def __init__(self): #Might alter parameters
        super(Model, self).__init__()
        model = nn.Sequential(torch.OrderedDict([
          ('conv1', torch.nn.Conv2d()),#Numbers
          ('BN1', torch.nn.BatchNorm2d()),
          ('activate1', torch.nn.ELU(alpha=1.0)),
          ('conv2', torch.nn.Conv2d()),#Numbers
          ('activate2', torch.nn.ELU(alpha=1.0))
          ('conv3', torch.nn.Conv2d()),#Numbers
          ('activate3', torch.nn.ELU(alpha=1.0))
        ]))
        #Insert hyperparameters here

    def call(self, x):
        return model(x)
