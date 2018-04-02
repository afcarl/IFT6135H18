import torch
from torch import nn
import torch.nn.functional as F
from ntm import NTM
from sequence_generator import *
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from IPython import display
import time
from mpl_toolkits.axes_grid1 import make_axes_locatable

def generate_sequences(nb_batches, max_len=10, mini_batch_size=10):
    # module = torch.cuda if cuda else torch
    #print(1)
    for batch_idx in range(nb_batches):
        # yield one batch
        T = np.random.randint(1, max_len + 1)
        #T = np.random.choice(list(range(1, max_len + 1)), 1, p=np.arange(1, max_len+1) * 2./((max_len+1) * (max_len)))[0]
        X = np.random.randint(0, 2, (mini_batch_size, T + 1, 9)).astype(float)
        X[:, :, -1] = np.array(T*[0]+[1])
        X[:, -1, :-1] = np.array(8 * [0])


        yield Variable(torch.from_numpy(X)).float()  
        
def generate_sequences_fixed_length(nb_batches, length=10, mini_batch_size=10):
    # module = torch.cuda if cuda else torch
    #print(1)
    for batch_idx in range(nb_batches):
        # yield one batch
        T = length
        X = np.random.randint(0, 2, (mini_batch_size, T + 1, 9)).astype(float)
        X[:, :, -1] = np.array(T*[0]+[1])
        X[:, -1, :-1] = np.array(8 * [0])


        yield Variable(torch.from_numpy(X)).float()  
        
        
class RNN(nn.Module):
    def __init__(self, input_size=9, hidden_size=100, output_size=9):
        super(RNN, self).__init__()
        MAX = 1024
        EOS = torch.from_numpy(np.array(8*[0] + [1])).float()
        self.hidden_size = hidden_size
        self.LSTM = nn.LSTMCell(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)
        self.activation = torch.nn.functional.sigmoid
        self.hidden_state0 = Parameter(torch.zeros(1, hidden_size)).float()
        self.cell_state0 = Parameter(torch.zeros(1, hidden_size)).float()
        self.zero_vector = Parameter(torch.zeros(MAX, 9)).float()
        #self.zero_vector = Parameter(EOS.expand(MAX, 9))
        
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
            hidden_state, cell_state, output = self.step(self.zero_vector[:N,:], hidden_state, cell_state)
            outputs.append(self.activation(output.unsqueeze(2).transpose(1, 2)))
        return torch.cat(outputs, 1)

if __name__ == '__main__':
    lstmk = torch.load('logs/13_39_55_vanillaLSTM_min_l=1_batch=200_lr=0.01/lstm.pkl')