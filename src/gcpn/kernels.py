# Feature functions for kernels
import torch

def linear(x):
    return x

def exp(x, alpha):
    return torch.exp(alpha*(x - 1.))

def d_exp(x, alpha):
    return alpha * exp(x, alpha)

def gauss(x, mu=0, sigma=1):
    return torch.exp((x-mu)**2/(sigma**2))
