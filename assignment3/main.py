import datetime

import torch
from ntm import NTM
from sequence_generator import *
from torch.autograd import Variable

tb_plot = True

if __name__ == "__main__":
    batch_size = 32
    dim = 8
    min_len = 1
    max_len = 20
    M = 10
    N = 32
    lr = 1e-4
    lstm = True

    cuda = True

    if tb_plot:
        from tensorboardX import SummaryWriter

        now = datetime.datetime.now()
        folder = (f'logs/{now.month}_{now.day}/'
                  f'{now.hour}_{now.minute}_{now.second}'
                  f'_{"LSTM" if lstm else "MLP"}_N={N}_M={M}'
                  f'_min_l={min_len}_batch={batch_size}_lr={lr}')
        print(folder)

        writer = SummaryWriter(log_dir=folder)

    seqgen = generate_inf_sequence(min_len, max_len, dim=dim, batch_size=batch_size, cuda=cuda)

    input_zero = Variable(torch.zeros(batch_size, dim + 1))
    ntm = NTM(N, M, dim + 1, dim, batch_size=batch_size, lstm=lstm)

    if cuda:
        input_zero = input_zero.cuda()
        ntm = ntm.cuda()

    criterion = torch.nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(ntm.parameters(), lr=lr)

    nb_samples = 0
    for step, inp, out in seqgen:

        nb_samples += batch_size

        ntm.reset(cuda)

        loss = 0
        acc = 0
        for i in range(inp.size(0)):
            ntm.send(inp[i])

        for i in range(inp.size(0) - 1):
            x = ntm.receive(input_zero)
            loss += criterion(x, out[i])
            acc += (x.round() == out[i]).float().mean()[0]

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
        opt.step()
        if nb_samples > 2000000:
            break
