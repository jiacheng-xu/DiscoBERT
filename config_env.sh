#!/usr/bin/env bash
# MS machines
cd /
sudo chmod 777 datadrive/
wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
sh Anaconda3-2019.03-Linux-x86_64.sh

##
##

sudo apt-get update
sudo apt-get install zip unzip

sudo apt-get install build-essential zsh
sh -c "$(curl -fsSL https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh)"

cd /datadrive/

git clone https://github.com/nlpyang/BertSum.git
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip
