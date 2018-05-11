#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --job-name=grad_g
#SBATCH --gres=gpu:1
# add following line if on cedar
##SBATCH --account=rpp-bengioy

source ~/.bashrc
source activate torch36


python my_dcgan_main.py --lanbda 1e-3 --penalty grad_g
