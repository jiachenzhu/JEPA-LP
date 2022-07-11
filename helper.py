import os
import math
from datetime import datetime

import torch
import torchvision

import matplotlib.pyplot as plt

import yaml

class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)

    def __repr__(self):
        return '%s' % str('\n'.join('%s : %s' % (k, repr(v)) for (k, v) in self.__dict__.items()))

def read_config(config_paths):
    final_config = {}
    for config_path in config_paths:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
            final_config = {**final_config, **config}
    
    return Struct(**final_config)

def save_checkpoint(checkpoint, directory, filename):
    os.makedirs(directory, exist_ok=True)
    torch.save(checkpoint, os.path.join(directory, filename))

class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def super_print(info, rank=None):
    if rank is None or rank == 0:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} {info}")

@torch.no_grad()
def visualize_kernel(kernels, directory, filename):
    # weight is a [channels, 3, kernel_size, kernel_size]
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()
    
    img = torchvision.utils.make_grid(kernels)
    plt.imshow(img.permute(1, 2, 0))
    
    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, filename), dpi=400)