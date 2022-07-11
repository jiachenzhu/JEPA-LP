import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP

class Encoder(nn.Module):
    def __init__(self, projector_type, projector_dims):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
        )
        self.projector = MLP(projector_type, projector_dims)

    def forward(self, x):
        return self.projector(self.backbone(x))
