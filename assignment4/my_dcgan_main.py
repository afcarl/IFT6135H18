import argparse
import datetime
import getpass
import os
import random

import matplotlib
import tensorboardX

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
#import torch.nn.utils.clip_grad_norm as clip_gradient
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torch.autograd import grad
from models import _netG, _netD, _netG_upsample
from utils import make_interpolation_noise

from inception_score import inception_score, mode_score

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='celebA',
                    help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', default='/data/lisa/data/celeba', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64,
                    help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='gan/logs/dcgan/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--mode', type=str, default='nsgan', metavar='N',
                    help='type of gan', choices=['mmgan', 'nsgan', 'lsgan', 'wgan'])
parser.add_argument('--upsample', type=str, default='convtranspose',
                    choices=['convtranspose', 'nearest', 'bilinear'],
                    help='Method used in the generator to upsample images.')
parser.add_argument('--name', type=str, default='', metavar='N',
                    help='name of the session')
parser.add_argument('--lanbda', type=float, default=.5, help='regularization')
parser.add_argument('--critic_iter', type=int, default=1, help='number of critic iterations')
parser.add_argument('--gen_iter', type=int, default=1, help='number of generator iterations')
parser.add_argument('--clip', type=float, default=.05, help='gradient clipping')

opt = parser.parse_args()
print(opt)

# CUDA
cudnn.benchmark = True
if torch.cuda.is_available():
    opt.cuda = True

# OUT FOLDER
opt.outf =  opt.outf
now = datetime.datetime.now()
opt.outf += opt.dataset + '/' + str(now.month) + '_' + str(now.day)
opt.outf += f'/{now.hour}_{now.minute}_{opt.mode}_{opt.name}'
opt.outf += f'_lambda={opt.lanbda}_citer={opt.critic_iter}_giter={opt.gen_iter}'
opt.outf += f'_beta1={opt.beta1}_upsample={opt.upsample}'

print('Outfile: ', opt.outf)
os.makedirs(opt.outf)
writer = tensorboardX.SummaryWriter(opt.outf)

# SEED
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

# DATA
print(f'Loading dataset {opt.dataset} at {opt.dataroot}')
# folder dataset
dataset = dset.ImageFolder(root=opt.dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.CenterCrop(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
print('Dataloader done')

# HYPER PARAMETERS
ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


def gradient_penaltyD(z, f):
    # gradient penalty
    z = Variable(z, requires_grad=True)
    o = f(z)
    g = grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True, retain_graph=True,
             only_inputs=True)[0]  # .view(z.size(0), -1)
    gp = ((g.view(z.size(0), -1).norm(p=2, dim=1)) ** 2).mean()
    return gp


def plot_images(tag, data, step, nrow=8):
    """Save mosaic of images to Tensorboard."""
    vutils.save_image(data,
                      opt.outf + '/' + tag + '_step_' + str(step) + '.png',
                      normalize=True, nrow=nrow)
    im = plt.imread(opt.outf + '/' + tag + '_step_' + str(step) + '.png')
    writer.add_image(tag, im, step)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# INITIALIZE MODELS
if opt.upsample == 'convtranspose':
    netG = _netG(ngpu)
else:
    netG = _netG_upsample(ngpu, opt.upsample)
netD = _netD(ngpu)
netD.apply(weights_init)
netG.apply(weights_init)

# Reload past models for a warm start
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

print(netD)
print(netG)

criterion = nn.BCEWithLogitsLoss()
if opt.mode == 'lsgan':
    criterion = nn.MSELoss()
if opt.mode == 'wgan':
    criterion = lambda out, target: ((1 - 2*target) * out).mean()

sigmoid = nn.Sigmoid()

# real images
input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

# input noise for the generator
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)

# input noise to plot samples
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)

# labels
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    if opt.mode != 'wgan':
        criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

step = 0
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader):
        step += 1
        for _ in range(opt.critic_iter):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            netD.zero_grad()
            real_cpu, _ = data
            batch_size = real_cpu.size(0)
            if opt.cuda:
                real_cpu = real_cpu.cuda()
            input.resize_as_(real_cpu).copy_(real_cpu)
            label.resize_(batch_size).fill_(real_label)
            inputv = Variable(input)
            labelv = Variable(label)

            output = netD(inputv).squeeze()
            errD_real = criterion(output, labelv)
            acc_real = sigmoid(output).data.round().mean()

            gp_real = gradient_penaltyD(inputv.data, netD)
            (opt.lanbda * gp_real).backward()

            errD_real.backward()
            f_x = output.data.mean()
            D_x = sigmoid(output).data.mean()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            labelv = Variable(label.fill_(fake_label))

            fake = netG(noisev)
            output = netD(fake.detach()).squeeze()
            errD_fake = criterion(output, labelv)

            gp_fake = gradient_penaltyD(fake.data, netD)
            (opt.lanbda * gp_fake).backward()

            errD_fake.backward()
            acc_fake = 1 - sigmoid(output).data.round().mean()
            f_G_z1 = output.data.mean()
            D_G_z1 = sigmoid(output).data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()

            if opt.mode == 'wgan':
                for param in netD.parameters():
                    param.data.clamp_(-opt.clip, opt.clip)

        for k in range(opt.gen_iter):
            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            if k > 0:
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noisev = Variable(noise)
                fake = netG(noisev)

            netG.zero_grad()
            output = netD(fake).squeeze()
            if opt.mode == 'nsgan':  # non-saturating gan
                # use the real labels (1) for generator cost
                labelv = Variable(label.fill_(real_label))
                errG = criterion(output, labelv)
            elif opt.mode == 'mmgan':  # minimax gan
                # use fake labels and opposite of criterion
                labelv = Variable(label.fill_(fake_label))
                errG = - criterion(output, labelv)
            elif opt.mode == 'lsgan':  # least square gan NOT WORKING
                # use real labels for generator
                labelv = Variable(label.fill_(real_label))
                errG = criterion(output, labelv)
            elif opt.mode == 'wgan':
                labelv = Variable(label.fill_(real_label))
                errG = criterion(output, labelv)

            errG.backward()
            f_G_z2 = output.data.mean()
            D_G_z2 = sigmoid(output).data.mean()
            optimizerG.step()

        if i % 50 == 0:
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.niter, i, len(dataloader),
                     errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            info = {'disc_cost': errD.data[0], \
                    'gen_cost': errG.data[0], \
                    'f_x': f_x,\
                    'f_G_z1': f_G_z1,\
                    'D_x': D_x, \
                    'D_G_z': D_G_z1, \
                    'acc_real': acc_real, \
                    'acc_fake': acc_fake, \
                    'logit_dist': f_x - f_G_z1, \
                    'penalty_real': opt.lanbda * (gp_real).data[0], \
                    'penalty_fake': opt.lanbda * (gp_fake).data[0], \
                    'gradient_norm_real': (gp_real).data[0], \
                    'gradient_norm_fake': (gp_fake).data[0]}

            for tag, val in info.items():
                writer.add_scalar(tag, val, global_step=step)

        if i % 50 == 0:
            plot_images('real_samples', real_cpu, step)
            fake = netG(fixed_noise)
            plot_images('fake_samples', fake.data, step)
            interpolation_noise = Variable(make_interpolation_noise(100, 64)).cuda()
            fake_interpol = netG(interpolation_noise)
            plot_images('fake_interpolation_samples', fake_interpol.data, step, nrow=10)

            # vutils.save_image(real_cpu,
            #         '%s/real_samples.png' % opt.outf,
            #         normalize=True)
            # vutils.save_image(fake.data,
            #         '%s/fake_samples_step_%03d.png' % (opt.outf, step),
            #         normalize=True)
            #
            # im = plt.imread('%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch))
            # writer.add_image('samples', im, step)
            #
            # for name, param in netD.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
            # for name, param in netG.named_parameters():
            #     writer.add_histogram(name, param.clone().cpu().data.numpy(), step)

        if step % 500 == 0:
            incep_score, _ = inception_score(fake.data, resize=True)
            md_score, _ = mode_score(fake.data, real_cpu, resize=True)
            print(f'Inception: {incep_score}')
            print(f'Mode score: {md_score}')
            writer.add_scalar('inception_score', incep_score, global_step=step)
            writer.add_scalar('mode_score', md_score, global_step=step)

    # do checkpointing
    if epoch % 5 == 0:
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
