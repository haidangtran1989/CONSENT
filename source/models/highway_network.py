import torch
import torch.nn as nn
from typing import Optional


class HighwayNetwork(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_layers: int,
                 activation: Optional[nn.Module] = None):
        super(HighwayNetwork, self).__init__()
        self.n_layers = n_layers
        self.nonlinear = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        self.gate = nn.ModuleList(
            [nn.Linear(input_dim, input_dim) for _ in range(n_layers)])
        for layer in self.gate:
            layer.bias = torch.nn.Parameter(0. * torch.ones_like(layer.bias))
        self.final_linear_layer = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU() if activation is None else activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer_idx in range(self.n_layers):
            gate_values = self.sigmoid(self.gate[layer_idx](inputs))
            nonlinear = self.activation(self.nonlinear[layer_idx](inputs))
            inputs = gate_values * nonlinear + (1. - gate_values) * inputs
        return self.final_linear_layer(inputs)
