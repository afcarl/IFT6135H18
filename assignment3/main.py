import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader
import ipdb
import numpy as np
from ntm import NTM
from sequence_generator import *
import datetime

tb_plot = True

batch_size = 32
dim = 8
min_len = 1
max_len = 20
M = 10
N = 32
lr = 1e-4

if tb_plot:
    from tensorboardX import SummaryWriter
    now = datetime.datetime.now()
    folder = f'/data/milatmp1/thomasva/DL_NTM/limited/logs/{now.day}_{now.month}/'
    name = f'{now.hour}_{now.minute}_{now.second}'
    name += f'_N={N}_M={M}_min_l={min_len}_batch={batch_size}_lr={lr}'
    print(name)

    writer = SummaryWriter(log_dir=folder+name)


seqgen = generate_inf_sequence(dim, min_len, max_len, batch_size=batch_size)


input_zero = Variable(torch.zeros(batch_size, dim+1)).cuda()


ntm = NTM(N, M, dim+1, dim+1, batch_size=batch_size)
ntm.cuda()

criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(ntm.parameters(), lr=1e-3)


for step, input in enumerate(seqgen):
    nb_samples = step*batch_size
    loss = 0
    ntm.reset()
    acc = 0
    for i in range(input.size(0)):
        ntm.send(input[i, :, :])

    for i in range(input.size(0)):
        #x = ntm.receive()
        x = ntm.receive(input_zero)
        loss += criterion(x, input[i, :, :])
        acc += (x.round() == input[i, :, :]).float().mean()[0]



    if step % 25 == 0:
        print('Step:', step)
        print('Loss:', loss.data[0]/input.size(0))
        print('Accuracy:', acc.data[0]/input.size(0))
        if tb_plot:
            writer.add_scalar('Loss', loss.data[0]/input.size(0), nb_samples)
            writer.add_scalar('Accuracy', acc.data[0]/input.size(0), nb_samples)

    opt.zero_grad()
    loss.backward()
    opt.step()
    if nb_samples > 2000000:
        break
