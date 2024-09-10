import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Encoder_Model import Encoder

class FeedForwardNN(nn.Module):
    def __init__(self, encoded_dim, hidden_dim, output_dim, n_layers_hidden=3, activation='relu'):
        super(FeedForwardNN, self).__init__()
        self.encoder = Encoder(in_c=1, z_dim=encoded_dim)
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(encoded_dim, hidden_dim))

        for _ in range(n_layers_hidden - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, output_dim))

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'elu':
            self.activation = F.elu
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
