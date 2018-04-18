import torch
import torch.nn.parallel
import torch.utils.data


def make_interpolation_noise(nz, batch_size, linear=True):
    num_interpol = 10
    z1 = torch.randn(8, nz)
    z2 = torch.randn(8, nz)
    noise = torch.zeros(8, num_interpol, nz)
    for i in range(num_interpol):
        p = (i + 1) / num_interpol
        noise[:, i] = p * z1 + (1 - p) * z2
    return noise.view(-1, nz, 1, 1)
