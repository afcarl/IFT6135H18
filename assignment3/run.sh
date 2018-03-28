#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
source activate torch36
export PYTHONUNBUFFERED=1
echo Running on $HOSTNAME
python main.py
