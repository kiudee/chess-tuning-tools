#! /bin/bash
# Image: nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

mkdir tuning
mkdir plots

apt update
apt install -y clang-6.0 gcc-8 g++-8 ninja-build pkg-config wget git libqtcore4 p7zip-full zlib1g-dev libpq-dev qt5-qmake qt5-default


wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $HOME/miniconda.sh
chmod +x miniconda.sh
bash ./miniconda.sh -b -p $HOME/miniconda
export PATH=$HOME/miniconda/bin:$PATH
echo PATH=$HOME/miniconda/bin:$PATH >> $HOME/.bashrc
rm $HOME/miniconda.sh
. $HOME/miniconda/etc/profile.d/conda.sh
#conda init bash
#source $HOME/.bashrc

conda create -y -n tuning python=3.8
conda activate tuning
pip install meson chess-tuning-tools


git clone https://github.com/LeelaChessZero/lc0.git
cd lc0 || exit
git checkout release/0.26
CC=clang-6.0 CXX=clang++-6.0 ./build.sh
cp build/release/lc0 $HOME/tuning/

cd $HOME || exit
git clone https://github.com/official-stockfish/Stockfish.git
cd $HOME/Stockfish/src || exit
make profile-build ARCH=x86-64-modern
cp stockfish $HOME/tuning/sf

cd $HOME || exit
git clone https://github.com/cutechess/cutechess.git
cd cutechess/projects || exit
qmake -after "SUBDIRS = lib cli"
make
export PATH=$HOME/cutechess/projects/cli:$PATH
echo PATH=$HOME/cutechess/projects/cli:$PATH >> $HOME/.bashrc

# Uncomment, if you need endgame tablebases:
## mkdir $HOME/syzygy
## cd $HOME/syzygy
## wget -e robots=off -r -np http://tablebase.sesse.net/syzygy/3-4-5/
## mv tablebase.sesse.net/syzygy/3-4-5/* .
## md5sum -c checksum.md5

# Insert your own download link for openings here:
cd $HOME/tuning || exit
wget https://cdn.discordapp.com/attachments/539960268982059008/723509619485442098/openings-dd.zip
7za x -oopenings openings-dd.zip

# Download the neural network to use:
wget https://training.lczero.org/get_network?sha=b30e742bcfd905815e0e7dbd4e1bafb41ade748f85d006b8e28758f1a3107ae3 -O 703810

# jupyter lab --ip=127.0.0.1 --port 8080 --no-browser --allow-root
