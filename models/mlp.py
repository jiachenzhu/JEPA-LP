from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, mlp_type, mlp_dims):
        super().__init__()
        layers = OrderedDict()
        type_count = {"l":0, "b": 0, "r": 0}
        dim_index = 0
        for layer_char in mlp_type:
            if layer_char == "l":
                type_count["l"] += 1
                layers["linear" + str(type_count["l"])] = nn.Conv2d(mlp_dims[dim_index], mlp_dims[dim_index + 1], kernel_size=1)
                dim_index += 1
            elif layer_char == "1":
                type_count["l"] += 1
                layers["linear" + str(type_count["l"])] = nn.Conv2d(mlp_dims[dim_index], mlp_dims[dim_index + 1], kernel_size=1, bias=False)
                dim_index += 1
            elif layer_char == "b":
                type_count["b"] += 1
                layers["bn" + str(type_count["b"])] = nn.BatchNorm2d(mlp_dims[dim_index])
            elif layer_char == "r":
                type_count["r"] += 1
                layers["relu" + str(type_count["r"])] = nn.ReLU()
            else:
                raise Exception(f"Unsupported layer: {layer_char}!")
        
        self.mlp = nn.Sequential(layers)

    def forward(self, x):
        return self.mlp(x)