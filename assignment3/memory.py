import math
import torch
from torch.autograd import Variable


class Memory:

    def __init__(self, N, M):
        self.memory = None
        self.memory_bias = Variable(
            torch.randn(1, N, M).cuda(), requires_grad=True) / math.sqrt(N)
