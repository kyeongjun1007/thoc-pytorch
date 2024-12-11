import torch
import numpy as np


def min_max_norm(x, dim=-1):
    if isinstance(x, np.ndarray):
        max_dist = x.max(axis=dim, keepdims=True)
        min_dist = x.min(axis=dim, keepdims=True)

    elif isinstance(x, torch.Tensor):
        max_dist, _ = torch.max(x, dim=dim, keepdim=True)
        min_dist, _ = torch.min(x, dim=dim, keepdim=True)

    else:
        raise TypeError(f"type of dist({type(x)}) is neither np.ndarray or torch.Tensor")

    diff = (max_dist - min_dist)
    out = (x - min_dist) / diff

    return out, (diff, min_dist)


def max_norm(x, dim=-1):
    if isinstance(x, np.ndarray):
        max_dist = x.max(axis=dim, keepdims=True)

    elif isinstance(x, torch.Tensor):
        max_dist, _ = torch.max(x, dim=dim, keepdim=True)

    else:
        raise TypeError(f"type of dist({type(x)}) is neither np.ndarray or torch.Tensor")


    out = (x / max_dist) + 0.2

    return out, (max_dist, 0)