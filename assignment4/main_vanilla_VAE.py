import numpy as np
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import argparse
from vanilla_VAE import VAE
from torch.utils.data.sampler import SubsetRandomSampler
import datetime

tb_plot = True

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 100)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate for the adam optimizer')
parser.add_argument('--tb-plot', action='store_false', default=True,
                    help='logs to tensorboard')
parser.add_argument('--loss-l', type=int, default=1, metavar='N',
                    help='Number of MC samples to evaluate the log-likelihood part of the loss')
parser.add_argument('--encoding-size', type=int, default=2, metavar='N',
                    help='Dimension of the encoding')
parser.add_argument('--notmnist', action='store_false', default=True,
                    help='if true then use MNIST')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if tb_plot:
    from tensorboardX import SummaryWriter

    now = datetime.datetime.now()
    folder = (f'logs/{now.month:0>2}_{now.day:0>2}/'
              f'{now.hour:0>2}_{now.minute:0>2}_{now.second:0>2}'
              f'_{"MNIST" if not args.notmnist else "CelebA"}_encoding={args.encoding_size}_batch={args.batch_size}'
              f'_lr={args.lr}_epochs={args.epochs}_L={args.loss_l}')

    print(folder)

    writer_train = SummaryWriter(log_dir=folder + '_train')
    writer_test = SummaryWriter(log_dir=folder + '_test')

if not args.notmnist:
    download_loc = "/u/lahlosal/data"

    transform = transforms.Compose([transforms.ToTensor(),
                                   ])

    mnist_train = datasets.MNIST(download_loc, train=True, transform=transform)
    mnist_test = datasets.MNIST(download_loc, train=False, transform=transform)

    train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=args.batch_size, shuffle=True)
else:
    datapath = "/u/lahlosal/celebA_small"

    transform = transforms.Compose([transforms.Resize(64),
                                    transforms.CenterCrop(64),
                                    transforms.ToTensor(),
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ])
    train_set = datasets.ImageFolder(root=datapath,
                                     transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    nb_train = int(0.8 * len(train_set))

    train_loader = torch.utils.data.DataLoader(
            train_set,
            sampler=SubsetRandomSampler(np.arange(nb_train)),
            batch_size=args.batch_size)

    test_loader = torch.utils.data.DataLoader(
            train_set,
            sampler=SubsetRandomSampler(np.arange(nb_train,len(train_set))),
            batch_size=args.batch_size)

for tr in train_loader:
    break
print(tr[0].shape)
print(len(train_loader.dataset))

vae = VAE(input_size=tr[0].shape[1] * tr[0].shape[2] * tr[0].shape[3],
          encoding_size=args.encoding_size)
optimizer = optim.Adam(vae.parameters(), lr=1e-3)
cuda = args.cuda
if cuda:
    vae = vae.cuda()


# assert 0 == 1


def train(epoch, print_stuff=False, loss_L=1):
    train_loss = 0
    train_nll = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data.view(-1,  tr[0].shape[1] * tr[0].shape[2] * tr[0].shape[3]))
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        loss, data_recon, nll = vae.loss(data, L=loss_L)
        loss.backward()
        train_loss += loss.data[0] * len(data)
        train_nll += nll.data[0] * len(data)
        optimizer.step()
        if print_stuff and batch_idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNLL: {:.4f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0], nll.data[0]))

    if args.tb_plot:
        writer_train.add_scalar('Loss',
                          train_loss / len(train_loader.dataset),
                          epoch)

    print('====> Epoch: {} Train loss: {:.4f}\tTrain nll: {:.4f}'.format(
        epoch, train_loss / len(train_loader.dataset), train_nll / len(train_loader.dataset)))

    return data, data_recon


def test(epoch, print_stuff=False):
    test_loss = 0
    test_nll = 0
    for batch_idx, (data, _) in enumerate(test_loader):
        data = Variable(data.view(-1, tr[0].shape[1] * tr[0].shape[2] * tr[0].shape[3]))
        if cuda:
            data = data.cuda()
        loss, data_recon, nll = vae.loss(data)
        test_loss += loss.data[0] * len(data)
        test_nll += nll.data[0] * len(data)
        if print_stuff and batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tNLL: {.4f}'.format(
                epoch, batch_idx * len(data), len(test_loader.dataset),
                100. * batch_idx / len(test_loader),
                loss.data[0], nll))
    if args.tb_plot:
        writer_test.add_scalar('Loss',
                          test_loss / len(test_loader.dataset),
                          epoch)

    print('====> Epoch: {} Test loss: {:.4f}\tTest nll: {:.4f}'.format(
        epoch, test_loss / len(test_loader.dataset), test_nll / len(test_loader.dataset)))

    return data, data_recon


def show(data, data_recon, out_file):
    fig, axes = plt.subplots(1, 2)
    img1 = make_grid(data.data.view(-1, tr[0].shape[1], tr[0].shape[2], tr[0].shape[3]), nrow=10).cpu()
    img2 = make_grid(data_recon.data.view(-1, tr[0].shape[1], tr[0].shape[2], tr[0].shape[3]), nrow=10).cpu()
    npimg1 = img1.numpy()
    npimg2 = img2.numpy()
    axes[0].imshow(np.transpose(npimg1, (1, 2, 0)), interpolation='nearest')
    axes[1].imshow(np.transpose(npimg2, (1, 2, 0)), interpolation='nearest')
    fig.savefig(out_file)


def save_sample(epoch):
    sample = Variable(torch.randn(64, args.encoding_size))
    if cuda:
        sample = sample.cuda()
    sample = vae.decode(sample).cpu()
    save_image(sample.data.view(64, tr[0].shape[1], tr[0].shape[2], tr[0].shape[3]),
               folder + '_train/sample_' + str(epoch) + '.png', )


save_sample(0)


for epoch in range(1, args.epochs + 1):
    data, data_recon = train(epoch, loss_L=args.loss_l)
    data_test, data_recon_test = test(epoch)
    if epoch % 10 == 0:
        show(data, data_recon, folder + '_train/train_{}.png'.format(epoch))
        show(data_test, data_recon_test, folder + '_train/test_{}.png'.format(epoch))
        torch.save(vae, folder + '_train/model.pkl')
        save_sample(epoch)
