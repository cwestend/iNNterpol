import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, hyperparameters):
        super(Network, self).__init__()

        # The hyperparameters of the network are saved for reproduction
        self.hyperparameters = hyperparameters
        
        self.input_size = hyperparameters['input_size']
        self.hidden_size = hyperparameters['hidden_size']
        self.n_hidden_layers = hyperparameters['n_hidden_layers']
        self.output_size = hyperparameters['output_size']
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

        """
        #self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        # 3 layer encoder
        self.layers.append(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2))
        self.layers.append(act)
        # Results 1D 16*kernels of length=35
        self.layers.append(nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2))
        self.layers.append(act)
        # Results 1D 32*kernels of length=17
        self.layers.append(nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2))
        # No need for nonlinear at the end?
        # self.layers.append(act)
        # Results 1D 64*kernels of length=8
        # Linear of 8? kernel of length 8?
        # The decoder
        self.layers.append(nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2))
        self.layers.append(act)
        self.layers.append(nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2))
        self.layers.append(act)
        self.layers.append(nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2))
        """
        
        """
        
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2), # 1D 16*kernels of length=35
            act,
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2), # 1D 32*kernels of length=17
            act,
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2) # 1D 64*kernels of length=8
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=3, stride=2),
            act,
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2), 
            act,
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=3, stride=2) 
        )
        """
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=128, kernel_size=3, stride=2, padding=1), # 1D 128 kernels of length=36
            act,
            #nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1), # 1D 4*32 kernels of length=18
            #act,
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1), # 1D 4*64 kernels of length=18
            act,
            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=18, stride=1), # 1D 128 kernels of length=1
            act,
            nn.Conv1d(in_channels=128, out_channels=71, kernel_size=1, stride=1) # NiN 71 channels of 1
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=71, out_channels=128, kernel_size=1, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=18, stride=1),
            act,
            nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1),
            act,
            #nn.ConvTranspose1d(in_channels=128, out_channels= 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            #act,
            nn.ConvTranspose1d(in_channels=128, out_channels=4, kernel_size=3, stride=2, padding=1)
        )

        #for i in range(self.n_hidden_layers):
        #    self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
        #    self.layers.append(act)

        #self.layers.append(nn.Linear(self.hidden_size, self.output_size))

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
        """
        
        for layer in self.layers:
            x = layer(x)

        return x
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
        