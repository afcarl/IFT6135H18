import imageio
import torch
from torchvision import utils as vutils


def make_interpolation_noise(nz, num_interpolation=10, num_interpolated=8):
    z = torch.randn(2 * num_interpolated, nz)
    noise = torch.zeros(num_interpolated,
                        num_interpolation, nz)
    for i in range(num_interpolation):
        p = i / (num_interpolation - 1)
        noise[:, i] = p * z[:num_interpolated] \
                      + (1 - p) * z[num_interpolated:]
    return noise.view(-1, nz, 1, 1), z.view(-1, nz, 1, 1)


def make_interpolation_samples(samples, num_interpolation=10):
    """Return images interpolated in the x space.

    samples should be a tensor of size (2*nb_interpolated)x3x64x64.
    """
    num_interpolated = int(samples.size(0) / 2)
    interpolation = torch.zeros(samples.size(0) // 2,
                                num_interpolation, 3, 64, 64)
    for i in range(num_interpolation):
        p = i / (num_interpolation - 1)
        interpolation[:, i] = p * samples[:num_interpolated] \
                              + (1 - p) * samples[num_interpolated:]
    return interpolation.view(-1, 3, 64, 64)


def plot_images(logdir, writer, tag, data, step, nrow=8):
    """Save mosaic of images to Tensorboard."""
    save_file = f'{logdir}/{tag}_step={step}.png'
    vutils.save_image(data, save_file,
                      normalize=True, nrow=nrow)
    im = imageio.imread(save_file)
    writer.add_image(tag, im, step)