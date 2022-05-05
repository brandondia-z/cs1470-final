import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Defines Model class
class Model(torch.nn.Module):
    def __init__(self, input_size): #Might alter parameters
        super(Model, self).__init__()

        self.input_size = input_size # (batch_size, 200, 24) => 1, 96, 1366
        self.hidden_size = 50
        self.channel_axis = 0
        self.freq_axis = 1
        self.time_axis = 2
        self.learning_rate = 5e-3

        # Input block
        self.zeroPad = torch.nn.ZeroPad2d(padding=(10, 10, 0, 0)) # (PadLeft, PadRight, PadTop, PadBottom)
        self.batchNorm0 = torch.nn.BatchNorm2d(num_features=3161) # TODO: Fix Params

        #Conv Block 1
        self.conv1 = torch.nn.Conv2d(in_channels=3161, out_channels=32, kernel_size=3, padding='same') #TODO: input param
        self.batchNorm1 = torch.nn.BatchNorm2d(num_features=32) # TODO: Fix Params
        self.elu1 = torch.nn.ELU(alpha=1.0)
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout1 = torch.nn.Dropout(p=0.1)

        #Conv Block 2
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same') #TODO: input param
        self.batchNorm2 = torch.nn.BatchNorm2d(num_features=64) # TODO: Fix Params
        self.elu2 = torch.nn.ELU(alpha=1.0)
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
        self.dropout2 = torch.nn.Dropout(p=0.1)

        #Conv Block 3
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same') #TODO: input param
        self.batchNorm3 = torch.nn.BatchNorm2d(num_features=128) # TODO: Fix Params
        self.elu3 = torch.nn.ELU(alpha=1.0)
        self.maxPool3 = torch.nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.dropout3 = torch.nn.Dropout(p=0.1)

        #Conv Block 4
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=200, kernel_size=3, padding='same') #TODO: input param
        self.batchNorm4 = torch.nn.BatchNorm2d(num_features=200) # TODO: Fix Params
        self.elu4 = torch.nn.ELU(alpha=1.0)
        self.maxPool4 = torch.nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        self.dropout4 = torch.nn.Dropout(p=0.1)

        # GRU block 1, 2, output
        self.GRU1 = torch.nn.GRU(input_size=200, hidden_size=128) #TODO: input param, add more params?
        self.GRU2 = torch.nn.GRU(input_size=128, hidden_size=64) #TODO: input param, add more params?
        self.dropout5 = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(in_features=64, out_features=self.hidden_size) #TODO: input param
        self.sigmoid = torch.nn.Sigmoid()

    def call(self, inputs):
        
        #Forward pass through pad/normalization
        zeroPad = self.zeroPad(inputs)
        batchNorm0 = self.batchNorm0(zeroPad)

        #Pass through Conv1
        conv1 = self.conv1(batchNorm0) #TODO: inputs, filters
        batchNorm1 = self.batchNorm1(conv1)
        elu1 = self.elu1(batchNorm1)
        maxPool1 = self.maxPool1(elu1)
        dropout1 = self.dropout1(maxPool1)
        
        #Pass through Conv2
        conv2 = self.conv2(dropout1)
        batchNorm2 = self.batchNorm2(conv2)
        elu2 = self.elu2(batchNorm2)
        maxPool2 = self.maxPool2(elu2)
        dropout2 = self.dropout2(maxPool2)

        #Pass through Conv3
        conv3 = self.conv3(dropout2)
        batchNorm3 = self.batchNorm3(conv3)
        elu3 = self.elu3(batchNorm3)
        maxPool3 = self.maxPool3(elu3)
        dropout3 = self.dropout3(maxPool3)

        #Pass through Conv4
        conv4 = self.conv4(dropout3)
        batchNorm4 = self.batchNorm4(conv4)
        elu4 = self.elu4(batchNorm4)
        maxPool4 = self.maxPool4(elu4)
        dropout4 = self.dropout4(maxPool4)

        #Reshape before GRU
        print(dropout4.shape)
        reshaped = torch.reshape(dropout4, (8, 200)) # TODO: Change shape

        #Pass through GRU layers and final forward pass through linear layer
        gru1, hiddenState1 = self.GRU1(reshaped)
        gru2, hiddenState2 = self.GRU2(gru1)
        gru_drop = self.dropout5(gru2)
        linear = self.linear(gru_drop)
        activated = self.sigmoid(linear)

        return activated
<<<<<<< HEAD
    
=======
>>>>>>> 32c6f53bcf4815471e7407c2475e00264c590b06
