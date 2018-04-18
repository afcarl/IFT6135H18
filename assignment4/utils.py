from math import sqrt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import grad


def make_interpolation_noise(nz, batch_size, linear=True):
    num_interpol = 10
    z1 = torch.randn(8, nz)
    z2 = torch.randn(8, nz)
    noise = torch.zeros(8, num_interpol, nz)
    for i in range(num_interpol):
        p = (i+1)/num_interpol 
        noise[:, i] = p * z1 + (1-p) * z2
    return noise.view(-1, nz, 1, 1)
