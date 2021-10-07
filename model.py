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
        self.layers.append(nn.Linear(self.input_size, self.hidden_size))
        self.layers.append(act)

        for i in range(self.n_hidden_layers):
            self.layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            self.layers.append(act)

        self.layers.append(nn.Linear(self.hidden_size, self.output_size))

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
        for layer in self.layers:
            x = layer(x)

        return x
    