import torch

def relative_l2_loss(pred, y):
    return (torch.norm(pred - y, dim=(-2, -1)) / torch.norm(y, dim=(-2, -1))).mean()