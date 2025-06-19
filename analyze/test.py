import torch.nn as nn
import torch
import numpy as np


def sample_mask(logits, tau=0.3):
    num_vars = len(logits)
    mask = gumbel_sigmoid(logits, tau=tau)
    non_diagonal_mask = torch.ones(num_vars, num_vars) - torch.eye(num_vars)
    # Set diagonal entries to 0
    mask = mask * non_diagonal_mask
    return mask


def sample_logistic(shape, out=None):
    U = out.resize_(shape).uniform_() if out is not None else torch.rand(shape)
    return torch.log(U) - torch.log(1-U)


def gumbel_sigmoid(logits, tau=1):
    dims = logits.dim()
    logistic_noise = sample_logistic(logits.size(), out=logits.data.new())
    y = logits + logistic_noise
    print(logistic_noise, y)
    return torch.sigmoid(y / tau)

input_dim = 3
logits = nn.Parameter(torch.zeros(input_dim, input_dim))
mask = sample_mask(logits, tau=0.3)

print(mask)