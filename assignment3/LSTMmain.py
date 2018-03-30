import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--lr")
parser.add_argument("--minibatchsize")

args = parser.parse_args()
lr = float(args.lr)
minibatch_size = int(args.minibatchsize)
n_sequences = min(20000, int(2e6/minibatch_size))

cuda = torch.cuda.is_available()


def generate_sequences(nb_batches, max_len=10, mini_batch_size=10):
    # module = torch.cuda if cuda else torch
    # print(1)
    for batch_idx in range(nb_batches):
        # yield one batch
        T = np.random.randint(1, max_len + 1)
        X = np.random.randint(0, 2, (mini_batch_size, T + 1, 9)).astype(float)
        X[:, :, -1] = np.array(T * [0] + [1])
        X[:, -1, :-1] = np.array(8 * [0])

        yield Variable(torch.from_numpy(X)).float()


def generate_sequences_fixed_length(nb_batches, length=10, mini_batch_size=10):
    # module = torch.cuda if cuda else torch
    # print(1)
    for batch_idx in range(nb_batches):
        # yield one batch
        T = length
        X = np.random.randint(0, 2, (mini_batch_size, T + 1, 9)).astype(float)
        X[:, :, -1] = np.array(T * [0] + [1])
        X[:, -1, :-1] = np.array(8 * [0])

        yield Variable(torch.from_numpy(X)).float()


class RNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=100, output_size=9):
        super(RNN, self).__init__()
        MAX = 1024
        EOS = torch.from_numpy(np.array(8 * [0] + [1])).float()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = functional.sigmoid
        self.hidden_state0 = Parameter(torch.zeros(1, hidden_size)).float()
        self.cell_state0 = Parameter(torch.zeros(1, hidden_size)).float()
        # self.zero_vector = Parameter(torch.zeros(MAX, 9)).float()
        self.zero_vector = Parameter(EOS.expand(MAX, 9))

    def step(self, input_vector, hidden_state, cell_state):
        hidden_state, cell_state = self.LSTM(input_vector, (hidden_state, cell_state))
        return hidden_state, cell_state, self.fc(hidden_state)

    def forward(self, input_vectors):
        N = input_vectors.shape[0]
        T = input_vectors.shape[1] - 1

        hidden_state = self.hidden_state0.expand(N, self.hidden_size)
        cell_state = self.cell_state0.expand(N, self.hidden_size)

        for t in range(T + 1):
            hidden_state, cell_state, _ = self.step(input_vectors[:, t, :], hidden_state, cell_state)

        outputs = []
        for t in range(T):
            hidden_state, cell_state, output = self.step(self.zero_vector[:N, :], hidden_state, cell_state)
            outputs.append(self.activation(output.unsqueeze(2).transpose(1, 2)))
        return torch.cat(outputs, 1)



criterion = torch.nn.BCELoss()
rnn = RNN()
optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
print_every = 25
if cuda:
    print('Using CUDA')
    rnn = rnn.cuda()
    # input_vectors = input_vectors2.cuda()
print("Training with lr = {0}, minibatch_size = {1}".format(lr, minibatch_size))
accuracies = []
running_losses = []

lossfct = functional.binary_cross_entropy

running_loss = 0.0

step = -1
for minibatch in generate_sequences(n_sequences, 20, minibatch_size):
    step += 1
    if cuda:
        minibatch = minibatch.cuda()
    optimizer.zero_grad()
    outputs = rnn(minibatch)
    loss = lossfct(outputs[:, :, :], minibatch[:, :-1, :], size_average=True)
    loss.backward()
    optimizer.step()
    running_loss = loss.data[0] / (minibatch.shape[1])
    if step % print_every == print_every - 1:
        list_input_vectors = [list(generate_sequences_fixed_length(1, j + 1, 1000))[0].cuda() for j in range(20)]
        accuracy_per_seqlength = []
        for x in list_input_vectors:
            accuracy_per_seqlength.append(
                torch.mean(((rnn(x[:, :, :])[:, :, :-1] > .5).float() == x[:, :-1, :-1]).float()).data[0])
        accuracy_approx = np.mean(accuracy_per_seqlength)
        running_losses.append(running_loss)

        accuracies.append(accuracy_approx)

        print(f'Step: {step + 1:<9}'
              f'Loss: {running_loss:<10.4f}'
              f'Accuracy: {accuracy_approx:<10.4f}'
              f'Accuracy 20: {accuracy_per_seqlength[-1]:<10.4f}')

plt.clf()
plt.subplot(1, 3, 1)
plt.plot(running_losses)
plt.ylabel("running_loss")
plt.subplot(1, 3, 2)
plt.plot(accuracies)
plt.ylabel("accuracy")
plt.subplot(1, 3, 3)
plt.plot(accuracy_per_seqlength)
plt.ylabel("accuracy per seqlength")
plt.gcf().set_size_inches((18, 6))
import pickle
pickle.dump({'losses': running_losses,
             'accuracies': accuracies,
             'accuracy_per_seqlength': accuracy_per_seqlength},
            open(f''))