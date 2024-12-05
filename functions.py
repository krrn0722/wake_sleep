import numpy as np
import torch

def sigmoid(x):
    return np.exp(np.minimum(x, 0)) / (1 + np.exp(- np.abs(x)))

def sigmoid_tensor(x):
    x = torch.clip(x, -500, 500)
    return 1 / (1 + torch.exp(-x))
    return torch.exp(torch.minimum(x, torch.zeros_like(x))) / (1 + torch.exp(- torch.abs(x)))