import random

import numpy
import torch


def generate_inf_sequence(min_len, max_len, cuda, dim=8, batch_size=1):
    """Generate batches of sequences of 8 bits ad infinitum.

    In fact there are 9 bits in the input because the last bit stands for the end of sentence.
    Yield the batch id, the input of size (T+1) * batch * 9 and the output of size T * batch * 8.

    The organization of each batch is similar to what is required by Pytorch's LSTM.
    The first axis index the sequence.
    The second axis index the batch.
    The third and last axis index the dimension of each sample.
    """

    batch_id = 0
    while True:
        batch_id += 1

        seq_len = random.randint(min_len, max_len)

        in_seq = create_sequence(seq_len, batch_size, cuda, dim=dim)
        out_seq = in_seq.clone()[:-1, :, :-1]

        yield batch_id, in_seq, out_seq


def create_sequence(seq_len, batch_size, cuda, dim=8):
    """Return batch_size sequences of a given length."""
    # Make the sequence longer by 1 on the 1st and 3rd dimensions to append the EOS
    seq = (numpy.random.rand(seq_len + 1, batch_size, dim + 1) > 0.5).astype(numpy.float32)

    # Define the End Of Sequence character
    seq[-1] = 0
    seq[:, :, -1] = 0
    seq[-1, :, -1] = 1

    seq = torch.autograd.Variable(torch.from_numpy(seq))
    if cuda:
        seq = seq.cuda()
    return seq
