import numpy as np
import torch.nn as nn


'''
Define a ResNet neural network
'''

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, out_channels):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Linear(out_channels, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.fc(x)
        # skip connection
        x += residual
        x = self.relu(x)

        return x


# ResNet
class ResNet(nn.Module):
    def __init__(self, block, nom, noj):
        super(ResNet, self).__init__()
        self.in_fc = nn.Linear(nom*3*2, 2048)
        self.relu = nn.ReLU()
        self.residual_layer = self.make_layer(block, 2048, 5)
        self.out_fc = nn.Linear(2048, noj*3*4)

    def make_layer(self, block, out_channels, nob):
        '''
        Get a layer with multiple residual blocks

        <Params>
        block: block class to create
        out_channels: number of output channels, dimension of the block will be (out_channels, out_channels)
        nob: number of blocks for this layer

        <Return>
        Sequence of blocks 

        '''

        layer = []
        for _ in range(nob):
            layer.append(block(out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.in_fc(x)
        x = self.relu(x)
        x = self.residual_layer(x)
        x = self.out_fc(x)
        return x

