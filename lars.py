import torch
import torch.nn as nn
import torch.nn.functional as F

def is_bias_or_norm(p):
    return p.ndim == 1

class LARS(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum, weight_decay, eta, weight_decay_filter=True, lars_adaptation_filter=True):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta,
                        weight_decay_filter=weight_decay_filter, lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            eta = group['eta']
            weight_decay_filter = group['weight_decay_filter']
            lars_adaptation_filter = group['lars_adaptation_filter']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    if not weight_decay_filter or not is_bias_or_norm(p):
                        d_p = d_p.add(p, alpha=weight_decay)

                if not lars_adaptation_filter or not is_bias_or_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(d_p)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0., (eta * param_norm / update_norm), one),
                                    one)
                    d_p = d_p.mul(q)

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                
                p.add_(d_p, alpha=-lr)