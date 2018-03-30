#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
source activate deep
for mb in 1 2 4 10 20 50 100 200 500 1000 2000
do
python ../LSTMmain.py --lr 1e-5 --minibatchsize $mb

done