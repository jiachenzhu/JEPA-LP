import torch
import torch.nn as nn
import torch.nn.functional as F

class Predictor(nn.Module):
    def __init__(self, representation_dim, latent_dim):
        super().__init__()
        self.predictor_dictionary = nn.Linear(latent_dim, representation_dim * representation_dim)

    def forward(self, initial_state, latent_variable, num_steps):
        # initial_state shape n l c
        # latent_variable shape n latent
        # predictor shape n l c c

        next_state_ground_true = initial_state[:, num_steps:]

        n, l, c = initial_state.shape
        predictor = self.predictor_dictionary(latent_variable).view(n, c, c)

        # predictions = []
        next_state = initial_state
        for i in range(num_steps):
            next_state = torch.einsum("nlc,ncb->nlb", next_state[:, :-num_steps], predictor)
            # predictions.append(next_state)

        return next_state, next_state_ground_true

