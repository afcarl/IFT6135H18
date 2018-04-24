#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --job-name=celeba
#SBATCH --output=celeba.out
#SBATCH --qos=high
#SBATCH --gres=gpu:1

source activate pytorch36

past_logs=/data/milatmp1/lepriolr/gan/logs/dcgan/celebA/4_19/20_42_nsgan__lambda=0.5_citer=1_giter=1_beta1=0.5_upsample=nearest/

python my_dcgan_main.py --lanbda 0.5  --upsample nearest --netG $past_logs/netG_epoch_15.pth --netD $past_logs/netD_epoch_15.pth