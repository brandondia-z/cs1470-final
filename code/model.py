import torch as torch
import numpy as np

# Creates Model class
class Model(torch.nn.Module):
    def __init__(self): #Might alter parameters
        super(Model, self).__init__()

    def call():
        model = nn.Sequential(torch.OrderedDict([
          ('conv1', torch.nn.Conv2d()),#Numbers
          ('BN1', torch.nn.BatchNorm2d()),
          ('activate1', torch.nn.ELU(alpha=1.0)),
          ('conv2', torch.nn.Conv2d()),#Numbers
          ('activate2', torch.nn.ELU(alpha=1.0))
          ('conv3', torch.nn.Conv2d()),#Numbers
          ('activate3', torch.nn.ELU(alpha=1.0))

        ]))
    
    def loss():

