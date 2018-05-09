"""Arguments parser."""
import argparse
import datetime
import getpass
import os
import random

import torch
from torch import cuda
from torch.backends import cudnn


def get_arguments():
    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument('--dataset', default='celebA', choices=['celebA'])
    parser.add_argument('--dataroot', default='/data/lisa/data/celeba',
                        help='path to dataset')
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to network')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64,
                        help='Size of the output image from the generator')
    parser.add_argument('--ndf', type=int, default=64,
                        help='Size of the input image to the discriminator')

    # Optimization
    parser.add_argument('--niter', type=int, default=150, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--manual-seed', type=int, help='manual seed')

    # GAN architecture
    parser.add_argument('--mode', type=str, default='mmgan', metavar='N',
                        help='Type of GAN: minimax, non-saturating, least-square, Wasserstein.',
                        choices=['mmgan', 'nsgan', 'lsgan', 'wgan'])
    parser.add_argument('--upsample', type=str, default='nearest',
                        choices=['convtranspose', 'nearest', 'bilinear'],
                        help='Method used in the generator to up-sample images.')
    parser.add_argument('--critic_iter', type=int, default=1,
                        help='number of critic iterations')
    parser.add_argument('--gen_iter', type=int, default=1,
                        help='number of generator iterations')
    parser.add_argument('--lanbda', type=float, default=1,
                        help='Regularization factor for the gradient penalty.')
    parser.add_argument('--penalty', type=str, default='both', choices=['real', 'fake', 'both', 'uniform', 'midinterpol', 'grad_g'],
                        help='Distribution on which to apply gradient penalty.')

    # Checkpoints
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--outf', default='gan/logs/dcgan/',
                        help='folder to output images and model checkpoints')
    parser.add_argument('--name', type=str, default='', metavar='N',
                        help='Name of the session appended to folder name.')

    opt = parser.parse_args()
    print(opt)

    # Number of input channels (RGB)
    opt.nc = 3

    # CUDA
    cudnn.benchmark = True
    opt.cuda = cuda.is_available()

    # SEED
    if opt.manual_seed is None:
        opt.seed = random.randint(1, 10000)
    else:
        opt.seed = opt.manual_seed
    print("Random Seed: ", opt.seed)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        cuda.manual_seed_all(opt.seed)

    # OUT FOLDER
    root_path = f'/data/milatmp1/{getpass.getuser()}'
    now = datetime.datetime.now()
    strpenalty = '0' if opt.lanbda <= 0 else f'{opt.lanbda}{opt.penalty}'
    folder_path = (
        f'{opt.dataset}/{now.month}_{now.day}'
        f'/{now.hour}_{now.minute}_{opt.mode}_{opt.name}'
        f'_gp={strpenalty}_citer={opt.critic_iter}_giter={opt.gen_iter}'
        f'_beta1={opt.beta1}_upsample={opt.upsample}_seed={opt.seed}'
    )
    opt.outf = os.path.join(root_path, opt.outf, folder_path)

    print('Outfile: ', opt.outf)
    os.makedirs(opt.outf)

    return opt
