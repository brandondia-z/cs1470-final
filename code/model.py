import torch as torch
import numpy as np

# Creates Model class
class Model(torch.nn.Module):
    def __init__(self, input_size, hidden_size): #Might alter parameters
        super(Model, self).__init__()

        self.input_size = input_size # 1366
        self.hidden_size = hidden_size # 50
        self.learning_rate = 5e-3
        self.optimizer = torch.optim.Adam(self.learning_rate)

        # Input block
        self.zeroPad = torch.nn.ZeroPad2d(padding=(37,37,0,0)) # (PadLeft, PadRight, PadTop, PadBottom)
        self.batchNorm1 = torch.nn.BatchNorm2d(num_features=1366) # TODO: Fix Params

        #Conv Block 1
        self.conv1 = torch.nn.functional.conv2d(input=1366, weight=64, stride=3, padding='same') #TODO: input param
        self.batchNorm1 = torch.nn.BatchNorm2d(num_features=1) # TODO: Fix Params
        self.elu1 = torch.nn.ELU(alpha=1.0)
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout1 = torch.nn.Dropout(p=0.1)

        #Conv Block 2
        self.conv2 = torch.nn.Conv2d(in_channels=1366, out_channels=128, kernel_size=3) #TODO: input param
        self.batchNorm2 = torch.nn.BatchNorm2d(num_features=1) # TODO: Fix Params
        self.elu2 = torch.nn.ELU(alpha=1.0)
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
        self.dropout2 = torch.nn.Dropout(p=0.1)

        #Conv Block 3
        self.conv3 = torch.nn.Conv2d(in_channels=1366, out_channels=128, kernel_size=3) #TODO: input param
        self.batchNorm3 = torch.nn.BatchNorm2d(num_features=1) # TODO: Fix Params
        self.elu3 = torch.nn.ELU(alpha=1.0)
        self.maxPool3 = torch.nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.dropout3 = torch.nn.Dropout(p=0.1)

        #Conv Block 4
        self.conv3 = torch.nn.Conv2d(in_channels=1366, out_channels=128, kernel_size=3) #TODO: input param
        self.batchNorm3 = torch.nn.BatchNorm2d(num_features=1) # TODO: Fix Params
        self.elu3 = torch.nn.ELU(alpha=1.0)
        self.maxPool3 = torch.nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.dropout3 = torch.nn.Dropout(p=0.1)

        # GRU block 1, 2, output
        self.GRU1 = torch.nn.GRU(input_size=1366, hidden_size=32) #TODO: input param, add more params?
        self.GRU2 = torch.nn.GRU(input_size=1366, hidden_size=32) #TODO: input param, add more params?
        self.dropout4 = torch.nn.Dropout(p=0.3)
        self.forward = torch.nn.Linear(in_features=1366, out_features=50) #TODO: input param



        # model = nn.Sequential(torch.OrderedDict([
        #   ('conv1', torch.nn.Conv2d()),#Numbers
        #   ('BN1', torch.nn.BatchNorm2d(axis=1, mode=2)),
        #   ('activate1', torch.nn.ELU(alpha=1.0)),
        #   ('MaxPool1', torch.nn.MaxPool2d((2, 4))),

        #   ('conv2', torch.nn.Conv2d()),#Numbers
        #   ('BN2', torch.nn.BatchNorm2d(axis=1, mode=2)),
        #   ('activate2', torch.nn.ELU(alpha=1.0)),
        #   ('MaxPool2', torch.nn.MaxPool2d((3, 4))),

        #   ('conv3', torch.nn.Conv2d()),#Numbers
        #   ('BN3', torch.nn.BatchNorm2d(axis=1, mode=2)),
        #   ('activate3', torch.nn.ELU(alpha=1.0))
        #   ('MaxPool3', torch.nn.MaxPool2d((2, 5))),

        #   ('Flatten', torch.nn.Flatten()),

        #   ('Dense', torch.nn.Linear()), #Params (in_feats, out_feats)
        #   ('DenseActivation', torch.nn.Sigmoid())

        # ]))


    def call():
        
        #Reshape after conv 4

    
    def loss():

