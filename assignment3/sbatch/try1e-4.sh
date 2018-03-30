#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
source activate deep
for lr in 0.00001 0.00005 0.0001 0.0005 0.001 0.005 0.01 0.05 0.1 0.5
do
for mb in 1 2 4 10 20 50 100 200 500 1000 2000
do
python ../LSTMmain.py --lr $lr --minibatchsize $mb
done
done