import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentGenerator(nn.Module):
    def __init__(self, representation_dim, hidden_dim, num_layers, latent_dim):
        super().__init__()
        self.gru = nn.GRU(representation_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        out, _ = self.gru(x)
        latent = self.fc(torch.relu(out[:, -1]))

        return latent

