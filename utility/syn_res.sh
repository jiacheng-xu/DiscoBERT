#!/usr/bin/env bash
read var1
rsync -rv --exclude='*.th'  --max-size=10m intern@20.36.20.141:/datadrive/msSum/$var1   /Users/jcxu/Downloads/
echo tensorboard --logdir /Users/jcxu/Downloads/$var1