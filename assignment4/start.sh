#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --job-name=celeba
#SBATCH --output=celeba.out
#SBATCH --qos=high
#SBATCH --gres=gpu:1

source activate pytorch36

python my_dcgan_main.py --lanbda 0.5  --upsample nearest