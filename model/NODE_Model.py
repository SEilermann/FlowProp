import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint
from model.Encoder_Model import Encoder

class ODE(nn.Module):
    """A simple n-layer feed forward neural network"""

    def __init__(self, dim_x, dim_z, hidden_size, n_layers_hidden=3, activation='relu'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(1+dim_x+dim_z, hidden_size))

        for _ in range(n_layers_hidden - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))

        self.layers.append(nn.Linear(hidden_size, dim_z))

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation =='elu':
            self.activation = F.elu
        else:
            raise ValueError("Unsupported activation function")

    def set_x(self, x):
        self.x = x

    def forward(self, t, z, x=None):
        if x is None:
            x = self.x
        if len(t.shape)<2:
            if len(t.shape)==0:
                t = t.unsqueeze(0).unsqueeze(1)
            t = t.repeat(x.shape[0], 1)
        temp=torch.cat([t,z,x],dim=-1)
        for layer in self.layers[:-1]:
            temp = self.activation(layer(temp))
        z = self.layers[-1](temp)
        return z


class NODE_Module(nn.Module):
    def __init__(self, dim_x, dim_z, hidden_size, n_layers_hidden=5, activation='elu', in_c=1, z_dim=32, solver="dopri5"):
        super().__init__()
        self.solver = solver
        self.encoder = Encoder(in_c, z_dim)
        self.ODE = ODE(dim_x, dim_z, hidden_size, n_layers_hidden, activation)

    def forward(self, t, z_0, x):
        x = self.encoder(x)
        self.ODE.set_x(x)
        t = t[0].squeeze(0) #odeint expects t to be 1D (as we expect all samples to be equally sampled we take the first sample)
        z = odeint(self.ODE, z_0, t, method=self.solver).permute(1, 0, 2)
        return z