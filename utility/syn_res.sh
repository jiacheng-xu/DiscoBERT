#!/usr/bin/env bash
read var1
rsync -rv --exclude='*.th'  --max-size=10m intern@52.229.11.179:/datadrive/DiscoBERT/$var1   /Users/jcxu/Downloads/
echo tensorboard --logdir /Users/jcxu/Downloads/$var1