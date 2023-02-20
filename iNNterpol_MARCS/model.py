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
        i = 0
        identity = self.layers[0](x)
        for layer in self.layers:
            # ResNet - adding weights skipping two layers 
            if (i % 4 == 0) and (i != 0 and i != len(self.layers)-1):
                x = layer(x) + identity
                identity = layer(x)
            else:
                x = layer(x)
            i = i + 1
        return x
    