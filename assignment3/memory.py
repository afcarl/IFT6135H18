import torch
from torch.autograd import Variable
import numpy as np


class Memory:

    def __init__(self, N, M):
        self.memory = None
        self.memory_bias = Variable(
            torch.randn(1, N, M).cuda(), requires_grad=True) / np.sqrt(N)
