import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from helper import AverageMeter
from einops import rearrange

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class FullGatherLayer(torch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """
    @staticmethod
    def forward(ctx, x):
        output = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = torch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

class VICRegLossModule:
    def __init__(self):
        self.loss_meter = AverageMeter()

    def __call__(self, representation, prediction_right, ground_true_right, prediction_left, ground_true_left, prediction_down, ground_true_down, prediction_up, ground_true_up, sim_coeff, std_coeff, cov_coeff):
        world_size = torch.distributed.get_world_size()

        repr_loss = F.mse_loss(prediction_right, ground_true_right.detach()) / 4 + \
                    F.mse_loss(prediction_left, ground_true_left.detach()) / 4 + \
                    F.mse_loss(prediction_down, ground_true_down.detach()) / 4 + \
                    F.mse_loss(prediction_up, ground_true_up.detach()) / 4

        x = rearrange(representation, "b h w c -> (b h w) c")

        x = torch.cat(FullGatherLayer.apply(x), dim=0)

        batch_size, num_features = x.shape
        
        x = x - x.mean(dim=0)

        std_x = torch.sqrt(x.var(dim=0) + 0.0001)
        std_loss = torch.mean(F.relu(1 - std_x))

        cov_x = (x.T @ x) / (batch_size - 1)
        cov_loss = 2 * off_diagonal(cov_x).pow_(2).sum().div(num_features)

        loss = sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

        self.loss_meter.update(loss.item())
        
        return loss

    def __str__(self):
        return  f"L/{self.loss_meter.avg:.4f}"

    def reset_meters(self):
        self.loss_meter.reset()