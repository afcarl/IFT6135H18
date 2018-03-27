import random

import numpy
import torch


def generate_sequence(nb_batch, max_len, batch_size=1, cuda=False):
    # module = torch.cuda if cuda else torch
    for batch_idx in range(nb_batch):

        seq_len = random.randint(2, max_len + 1)

        seq = (numpy.random.rand(seq_len, batch_size, 9) > 0.5).astype(numpy.uint8)

        seq[-1] = 0
        seq[:, -1] = 0
        seq[-1, -1] = 1

        seq = torch.autograd.Variable(torch.from_numpy(seq))

        yield seq
