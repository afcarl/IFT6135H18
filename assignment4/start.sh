#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --job-name=uniform1
#SBATCH --output=celeba.out
#SBATCH --gres=gpu:1
# add following line if on cedar
##SBATCH --account=rpp-bengioy

source activate pytorch36


python my_dcgan_main.py --lanbda 0 --penalty uniform
