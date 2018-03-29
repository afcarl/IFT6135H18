import datetime

import torch
import ipdb
from ntm import NTM
from sequence_generator import *
from torch.autograd import Variable

tb_plot = True

if __name__ == "__main__":
    batch_size = 1
    dim = 8
    min_len = 1
    max_len = 20
    M = 20
    N = 128
    lr = 1e-4
    use_cuda = False

    if tb_plot:
        from tensorboardX import SummaryWriter

        now = datetime.datetime.now()
        folder = (f'logs/{now.month}_{now.day}/'
                  f'{now.hour}_{now.minute}_{now.second}'
                  f'_N={N}_M={M}_min_l={min_len}_batch={batch_size}_lr={lr}')
        print(folder)

        writer = SummaryWriter(log_dir=folder)

    ntm = NTM(N, M, dim + 1, dim + 1, batch_size=batch_size)
    input_zero = Variable(torch.zeros(batch_size, dim + 1))
    if torch.cuda.is_available():
        use_cuda = True
        input_zero.cuda()
        ntm.cuda()

    seqgen = generate_inf_sequence(min_len, max_len, dim=dim, batch_size=batch_size, cuda=use_cuda)
    criterion = torch.nn.BCELoss()
    opt = torch.optim.Adam(ntm.parameters(), lr=lr)

    nb_samples = 0
    for step, inp, out in seqgen:
        nb_samples += batch_size
        loss = 0
        ntm.reset()
        acc = 0
        for i in range(inp.size(0)):
            ntm.send(inp[i])

        for i in range(inp.size(0) - 1):
            x = ntm.receive(input_zero)
            loss += criterion(x[:, :-1], out[i])
            acc += (x[:, :-1].round() == out[i]).float().mean()[0]

        meanloss = loss.data[0] / out.size(0)
        meanacc = acc.data[0] / out.size(0)

        if step % 25 == 0:
            print(f'Step: {step:<9}'
                  f'Loss: {meanloss:<10.4f}'
                  f'Accuracy: {meanacc:<10.4f}'
                  f'Length: {out.size(0):<5}')
            if tb_plot:
                writer.add_scalar('Loss', meanloss, nb_samples)
                writer.add_scalar('Accuracy', meanacc, nb_samples)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(ntm.parameters(), 10)
        opt.step()
        if nb_samples > 2000000:
            break
