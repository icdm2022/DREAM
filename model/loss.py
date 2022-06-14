import torch.nn as nn


def CrossEntropyLoss(output, target):
    cr = nn.CrossEntropyLoss()
    return cr(output, target)
