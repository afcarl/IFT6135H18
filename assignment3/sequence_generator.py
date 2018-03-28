import random

import numpy
import torch
import ipdb


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

def generate_inf_sequence(dim, min_len, max_len, batch_size=1, cuda=True):
    while True:
        seq_len = random.randint(min_len, max_len)

        # Make the sequence longer by 1 to append the EOS
        seq = (numpy.random.rand(seq_len+1, batch_size, dim+1) > 0.5).astype(numpy.float32)

        # Define the End Of Sequence character
        seq[-1] = 0
        seq[:, -1] = 0
        seq[-1, -1] = 1

        in_seq = torch.autograd.Variable(torch.from_numpy(seq))
        #out_seq = torch.autograd.Variable(torch.from_numpy(seq[:-1, :-1]))

        if cuda:
            yield in_seq.cuda()#, out_seq.cuda()
        else:
            yield in_seq#, out_seq
