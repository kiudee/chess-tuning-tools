#! /bin/bash
# Image: nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

mkdir tuning

apt update
apt install -y clang-6.0 gcc-8 g++-8 ninja-build pkg-config wget git libqtcore4 p7zip-full zlib1g-dev libpq-dev qt5-qmake qt5-default


wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
chmod +x miniconda.sh
bash ./miniconda.sh -b -p $HOME/miniconda
export PATH=~/miniconda/bin:$PATH
rm ~/miniconda.sh
. ~/miniconda/etc/profile.d/conda.sh
#conda init bash
#source ~/.bashrc

conda create -y -n tuning python=3
conda activate tuning
pip install meson chess-tuning-tools scikit-learn==0.22.2 jupyterlab


git clone https://github.com/LeelaChessZero/lc0.git
cd lc0 || exit
git checkout release/0.26
CC=clang-6.0 CXX=clang++-6.0 ./build.sh
cp build/release/lc0 ~/tuning/

cd ~ || exit
git clone https://github.com/official-stockfish/Stockfish.git
cd ~/Stockfish/src || exit
make profile-build ARCH=x86-64-modern
cp stockfish ~/tuning/sf

cd ~ || exit
git clone https://github.com/cutechess/cutechess.git
cd cutechess/projects || exit
qmake -after "SUBDIRS = lib cli"
make
export PATH=$HOME/cutechess/projects/cli:$PATH

# Uncomment, if you need endgame tablebases:
## mkdir $HOME/syzygy
## cd ~/syzygy
## wget -e robots=off -r -np http://tablebase.sesse.net/syzygy/3-4-5/
## mv tablebase.sesse.net/syzygy/3-4-5/* .
## md5sum -c checksum.md5

# Insert your own download link for openings here:
cd ~/tuning || exit
wget https://cdn.discordapp.com/attachments/539960268982059008/723509619485442098/openings-dd.zip
7za x -oopenings openings-dd.zip

# Download the neural network to use:
wget http://training.lczero.org/get_network?sha=50258149f09aa4469e703dfe6bbf314b5fc7b687c5f3146587aba11f67c54284 -O 703556

# jupyter lab --ip=127.0.0.1 --port 8080 --no-browser --allow-root
