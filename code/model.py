import torch
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

# Defines Model class
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.hidden_size = 50
        self.channel_axis = 0
        self.freq_axis = 1
        self.time_axis = 2
        self.learning_rate = 5e-3
        self.batch_size = 100

        # Input block
        self.zeroPad = torch.nn.ZeroPad2d(padding=(10, 10, 0, 0)) # (PadLeft, PadRight, PadTop, PadBottom)
        self.batchNorm0 = torch.nn.BatchNorm2d(num_features=1)

        #Conv Block 1
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.batchNorm1 = torch.nn.BatchNorm2d(num_features=32)
        self.elu1 = torch.nn.ELU(alpha=1.0)
        self.maxPool1 = torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))
        self.dropout1 = torch.nn.Dropout(p=0.1)

        #Conv Block 2
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.batchNorm2 = torch.nn.BatchNorm2d(num_features=64)
        self.elu2 = torch.nn.ELU(alpha=1.0)
        self.maxPool2 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(3,3))
        self.dropout2 = torch.nn.Dropout(p=0.1)

        #Conv Block 3
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding='same')
        self.batchNorm3 = torch.nn.BatchNorm2d(num_features=128)
        self.elu3 = torch.nn.ELU(alpha=1.0)
        self.maxPool3 = torch.nn.MaxPool2d(kernel_size=(4,4), stride=(4,4))
        self.dropout3 = torch.nn.Dropout(p=0.1)

        #Conv Block 4
        self.conv4 = torch.nn.Conv2d(in_channels=128, out_channels=200, kernel_size=3, padding='same')
        self.batchNorm4 = torch.nn.BatchNorm2d(num_features=200)
        self.elu4 = torch.nn.ELU(alpha=1.0)
        self.maxPool4 = torch.nn.MaxPool2d(kernel_size=(1,1), stride=(1,1))
        self.dropout4 = torch.nn.Dropout(p=0.1)

        # GRU block 1, 2, output
        self.GRU1 = torch.nn.GRU(input_size=1600, hidden_size=128)
        self.GRU2 = torch.nn.GRU(input_size=128, hidden_size=64)
        self.dropout5 = torch.nn.Dropout(p=0.3)
        self.linear = torch.nn.Linear(in_features=64, out_features=self.hidden_size)
        self.sigmoid = torch.nn.Sigmoid()

    def call(self, inputs):
        
        #Forward pass through pad/normalization
        zeroPad = self.zeroPad(inputs)
        batchNorm0 = self.batchNorm0(zeroPad)

        #Pass through Conv1
        conv1 = self.conv1(batchNorm0)
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
        flat= torch.flatten(dropout4, start_dim=1)

        #Pass through GRU layers and final forward pass through linear layer
        gru1, hiddenState1 = self.GRU1(flat)
        gru2, hiddenState2 = self.GRU2(gru1)
        gru_drop = self.dropout5(gru2)
        linear = self.linear(gru_drop)
        activated = self.sigmoid(linear)

        return activated
    
