#!/usr/bin/env bash
# MS machines
cd /
sudo chmod 777 datadrive/
wget https://repo.anaconda.com/archive/Anaconda2-2019.03-Linux-x86_64.sh
sh Anaconda2-2019.03-Linux-x86_64.sh
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
git clone https://github.com/PKU-TANGENT/NeuralEDUSeg.git
wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
unzip stanford-corenlp-full-2018-10-05.zip

export CLASSPATH=/home/cc/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
export CLASSPATH=/datadrive/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar
conda install cudatoolkit
conda install cudnn

sudo apt-get install default-jdk

cd /home/intern
git clone https://github.com/jiacheng-xu/pythonrouge.git
cd pythonrouge/
python setup.py install
cd pythonrouge/RELEASE-1.5.5/data/
rm WordNet-2.0.exc.db # only if exist
cd WordNet-2.0-Exceptions
rm WordNet-2.0.exc.db # only if exist
./buildExeptionDB.pl . exc WordNet-2.0.exc.db
cd ../
ln -s WordNet-2.0-Exceptions/WordNet-2.0.exc.db WordNet-2.0.exc.db
cd ../../../..

pyrouge_set_rouge_path /home/intern/pythonrouge/pythonrouge/RELEASE-1.5.5
sudo apt-get install libxml-parser-perl
python -m spacy download en
sudo apt-get install htop

conda install -c dglteam dgl-cuda10.0