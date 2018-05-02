"""Arguments parser."""
import argparse
import datetime
import getpass
import os
import random

import torch


def get_arguments():
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
    parser.add_argument('--manual-seed', type=int, help='manual seed')
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
    torch.backends.cudnn.benchmark = True
    opt.cuda = torch.cuda.is_available()

    # SEED
    if opt.manual_seed is None:
        opt.seed = random.randint(1, 10000)
    else:
        opt.seed = opt.manual_seed
    print("Random Seed: ", opt.seed)

    random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.seed)

    # OUT FOLDER
    opt.outf = f'/data/milatmp1/{getpass.getuser()}/' + opt.outf
    now = datetime.datetime.now()
    opt.outf += opt.dataset + '/' + str(now.month) + '_' + str(now.day)
    opt.outf += f'/{now.hour}_{now.minute}_{opt.mode}_{opt.name}'
    opt.outf += f'_lambda={opt.lanbda}_citer={opt.critic_iter}_giter={opt.gen_iter}'
    opt.outf += f'_beta1={opt.beta1}_upsample={opt.upsample}_seed={opt.seed}'

    print('Outfile: ', opt.outf)
    os.makedirs(opt.outf)
