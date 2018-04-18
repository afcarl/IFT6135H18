import torch
from torch.autograd import Variable
import ntm

dim = 8
input = Variable(torch.bernoulli(torch.rand(32, 20, dim + 1))).cuda()
input[:, :, -1] = 0
input[:, -1, -1] = 1

input_zero = Variable(torch.zeros(32, dim + 1)).cuda()

M = 64
N = 512

ntm = ntm.NTM(N, M, dim + 1, dim + 1, batch_size=32)
ntm.cuda()

criterion = torch.nn.BCELoss()
opt = torch.optim.Adam(ntm.parameters(), lr=1e-4)

for epoch in range(10000):
    loss = 0
    ntm.reset()
    acc = 0
    for i in range(20):
        ntm.send(input[:, i, :])

    for i in range(20):
        # x = ntm.receive()
        x = ntm.receive(input_zero)
        loss += criterion(x, input[:, i, :])
        acc += (x.round() == input[:, i, :]).float().mean()[0]

    print('Loss:', loss.data[0] / 20)
    print('Accuracy:', acc.data[0] / 20)
    opt.zero_grad()
    loss.backward()
    opt.step()
