import os
import matplotlib.pyplot as plt
from scipy.misc import imresize

# root path depends on your computer
root = '/u/lahlosal/CelebAdata/'
save_root = '/u/lahlosal/celebA/'
resize_size = 64

if not os.path.isdir(save_root):
    os.mkdir(save_root)
if not os.path.isdir(save_root):
    os.mkdir(save_root)
img_list = os.listdir(root)
save_img_list = os.listdir(save_root)

# ten_percent = len(img_list) // 10

for i in range(len(img_list)):
    if img_list[i] not in save_img_list:
        img = plt.imread(root + img_list[i])
        img = imresize(img, (resize_size, resize_size))
        plt.imsave(fname=save_root + img_list[i], arr=img)

    if (i % 1000) == 0:
        print('%d images complete' % i)