# 
# Model for the CAE used for MARCS data (input of 4x56, bottleneck of 71).

import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters
        
        self.activation = hyperparameters['activation']

        if (self.activation == 'relu'):
            act = nn.ReLU()
        if (self.activation == 'elu'):
            act = nn.ELU()
        if (self.activation == 'leakyrelu'):
            act = nn.LeakyReLU(0.2)

        # Define the layers of the network. We use a ModuleList here
        self.layers = nn.ModuleList([])
        
        # Start appending all layers
        # 1D Convolutional Auto-Encoder

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=3, stride=2, padding=1), # 1D 128 kernels of length=36
            act,
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # 1D 4*64 kernels of length=18
            act,
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=14, stride=1), # 1D 128 kernels of length=1
            act,
            nn.Conv1d(in_channels=128, out_channels=71, kernel_size=1, stride=1) # NiN 71 channels of length=1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=71, out_channels=128, kernel_size=1, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=14, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            act,
            nn.ConvTranspose1d(in_channels=128, out_channels=4, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

    def forward(self, x):
        """
        Evaluate the network

        Parameters
        ----------
        x : tensor
            Input tensor

        Returns
        -------
        tensor
            Output tensor
        """       
        
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        