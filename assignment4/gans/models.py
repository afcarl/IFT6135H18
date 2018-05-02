import torch
import torch.nn as nn


class _netG(nn.Module):
    """Generator network with square layers of  size [1, 4, 8, 16, 32]"""

    def __init__(self, opt):
        super(_netG, self).__init__()
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
                nn.Upsample(scale_factor=2, mode=opt.mode),
                nn.Conv2d(opt.ngf * 8, opt.ngf * 4, 5, padding=2, bias=False),
                nn.BatchNorm2d(opt.ngf * 4),
                nn.ReLU(True),
                # state size. (ngf*4) x 8 x 8
                nn.Upsample(scale_factor=2, mode=opt.mode),
                nn.Conv2d(opt.ngf * 4, opt.ngf * 2, 5, padding=2, bias=False),
                nn.BatchNorm2d(opt.ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.Upsample(scale_factor=2, mode=opt.mode),
                nn.Conv2d(opt.ngf * 2, opt.ngf, 5, padding=2, bias=False),
                nn.BatchNorm2d(opt.ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.Upsample(scale_factor=2, mode=opt.mode),
                nn.Conv2d(opt.ngf, opt.nc, 5, padding=2, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    def __init__(self, opt):
        super(_netD, self).__init__()
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

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)
