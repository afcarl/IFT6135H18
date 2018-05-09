import torch
import torch.nn as nn
from torch.autograd import Variable


class GeneratorNet(nn.Module):
    """Generator network with square layers of  size [1, 4, 8, 16, 32]"""

    def __init__(self, opt):
        super(GeneratorNet, self).__init__()
        self.ngpu = opt.ngpu

        if opt.upsample == 'convtranspose':
            self.main = nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(opt.ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d(opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(opt.ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(opt.ngf, opt.nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        else:
            self.main = nn.Sequential(
                # input is Z, going into a convolution.
                # NO need for upsampling here as it starts from dimension 1.
                nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(opt.ngf * 8),
                nn.ReLU(True),
                # state size. (ngf*8) x 4 x 4
                nn.Upsample(scale_factor=2, mode=opt.upsample),
                nn.Conv2d(opt.ngf * 8, opt.ngf * 4, 5, padding=2, bias=False),
                nn.BatchNorm2d(opt.ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(scale_factor=2, mode=opt.upsample),
                nn.Conv2d(opt.ngf * 4, opt.ngf * 2, 5, padding=2, bias=False),
                nn.BatchNorm2d(opt.ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.Upsample(scale_factor=2, mode=opt.upsample),
                nn.Conv2d(opt.ngf * 2, opt.ngf, 5, padding=2, bias=False),
                nn.BatchNorm2d(opt.ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.Upsample(scale_factor=2, mode=opt.upsample),
                nn.Conv2d(opt.ngf, opt.nc, 5, padding=2, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

        self.apply(weights_init)

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class DiscriminatorNet(nn.Module):
    def __init__(self, opt):
        super(DiscriminatorNet, self).__init__()
        self.ngpu = opt.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(opt.nc, opt.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(opt.ndf, opt.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(opt.ndf * 2, opt.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(opt.ndf * 4, opt.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(opt.ndf * 8, 1, 4, 1, 0, bias=False),
            # nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, inp):
        if isinstance(inp.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, inp, range(self.ngpu))
        else:
            output = self.main(inp)

        return output.view(-1, 1).squeeze(1)

    def gradient_penalty(self, inp):
        inp = Variable(inp, requires_grad=True)
        o = self.forward(inp)
        g = torch.autograd.grad(
            o, inp,
            grad_outputs=torch.ones_like(o),
            create_graph=True,
            only_inputs=True  # don't accumulate other gradients in .grad
        )[0]
        gp = ((g.view(inp.size(0), -1).norm(p=2, dim=1)) ** 2).mean()
        return gp

    def gradient_penalty_g(self, z, netG):
        #inp = Variable(inp, requires_grad=True)
        o = self.forward(netG(z))
        g = torch.autograd.grad(
            o, netG.parameters(),
            grad_outputs=torch.ones_like(o),
            create_graph=True,
            only_inputs=True  # don't accumulate other gradients in .grad
        )[0]
        gp = 0
        for grad in g:
            gp += ((grad.view(z.size(0), -1).norm(p=2, dim=1)) ** 2).mean()
        return gp


def weights_init(m):
    """Custom weights initialization."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
