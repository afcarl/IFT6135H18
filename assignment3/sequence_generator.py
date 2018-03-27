import random

import numpy
import torch


def generate_sequence(nb_batch, max_len, batch_size=1, cuda=False):
    # module = torch.cuda if cuda else torch
    for batch_idx in range(nb_batch):
        seq_len = random.randint(1, max_len)

        # Make the sequence longer by 1 to append the EOS
        seq = (numpy.random.rand(seq_len+1, batch_size, 9) > 0.5).astype(numpy.float32)

        # Define the End Of Sequence character
        seq[-1] = 0
        seq[:, -1] = 0
        seq[-1, -1] = 1

        in_seq = torch.autograd.Variable(torch.from_numpy(seq))
        out_seq = torch.autograd.Variable(torch.from_numpy(seq[:-1, :-1]))

        yield in_seq, out_seq
