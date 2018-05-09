#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --job-name=nogp
#SBATCH --gres=gpu:1
# add following line if on cedar
## SBATCH --account=rpp-bengioy

export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
source activate pytorch36
export PYTHONUNBUFFERED=1

echo Running on $HOSTNAME

python my_dcgan_main.py --lanbda 1 --penalty real
