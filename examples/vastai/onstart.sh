#! /bin/bash
# Image: nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04


if [ ! -f "$HOME"/tuning/config.json ]; then
    mkdir tuning
    mkdir tuning/plots

    # Install all the needed system packages needed for
    #  * Compilation of LeelaChessZero and Stockfish
    #  * Compilation of cutechess-cli
    #  * Checking out git repositories
    #  * Unpacking archives
    apt update
    apt install -y clang-6.0 gcc-8 g++-8 ninja-build pkg-config wget git libqtcore4 p7zip-full zlib1g-dev libpq-dev qt5-qmake qt5-default

    # We install the chess-tuning-tools in a minimal python installation.
    # With this we can avoid depending on the underlying system python
    # installation.
    wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O "$HOME"/miniconda.sh
    chmod +x miniconda.sh
    bash ./miniconda.sh -b -p "$HOME"/miniconda
    export PATH=$HOME/miniconda/bin:$PATH
    echo export PATH="$HOME"/miniconda/bin:"$PATH" >> "$HOME"/.bashrc
    rm "$HOME"/miniconda.sh
    . "$HOME"/miniconda/etc/profile.d/conda.sh

    # Set up the environment for the chess-tuning-tools:
    conda create -y -n tuning python=3.8
    conda activate tuning
    pip install meson chess-tuning-tools scikit-learn==0.22.2

    # Download a recent version of LeelaChessZero, compile it and copy it to the
    # tuning folder:
    git clone https://github.com/LeelaChessZero/lc0.git
    cd lc0 || exit
    git checkout release/0.26
    CC=clang-6.0 CXX=clang++-6.0 ./build.sh
    cp build/release/lc0 "$HOME"/tuning/

    # Download a recent version of Stockfish, compile it and copy it to the
    # tuning folder:
    cd "$HOME" || exit
    git clone https://github.com/official-stockfish/Stockfish.git
    cd "$HOME"/Stockfish/src || exit
    make profile-build ARCH=x86-64-modern
    make net
    cp stockfish "$HOME"/tuning/sf
    mv ./*.nnue "$HOME"/tuning

    # Download and compile cutechess-cli with minimal dependencies:
    cd "$HOME" || exit
    git clone https://github.com/cutechess/cutechess.git
    cd cutechess/projects || exit
    qmake -after "SUBDIRS = lib cli"
    make
    export PATH=$HOME/cutechess/projects/cli:$PATH
    echo export PATH="$HOME"/cutechess/projects/cli:"$PATH" >> "$HOME"/.bashrc

    ## Uncomment, if you need endgame tablebases. Remember to also activate it
    ## in the tuning script and for the engines themselves. Specifically, add
    ## "SyzygyPath": "/root/syzygy" to the "fixed_parameters" sections and set
    ## "adjudicate_tb": true and "tb_path": "/root/syzygy" in config.json.
    # mkdir $HOME/syzygy
    # cd $HOME/syzygy
    # wget -e robots=off -r -np http://tablebase.sesse.net/syzygy/3-4-5/
    # mv tablebase.sesse.net/syzygy/3-4-5/* .
    # md5sum -c checksum.md5

    # Insert your own download link for openings here:
    cd "$HOME"/tuning || exit
    wget https://cdn.discordapp.com/attachments/539960268982059008/723509619485442098/openings-dd.zip
    7za x -oopenings openings-dd.zip

    # Download the actual tuning configuration (config.json):
    wget https://gist.githubusercontent.com/kiudee/0584405676d6f71ca3dc6dfd02061663/raw/4a013b4d3700720f81326fe9c10254414dbca0f0/config.json

    # Download the neural network to use for LeelaChessZero:
    wget https://training.lczero.org/get_network?sha=b30e742bcfd905815e0e7dbd4e1bafb41ade748f85d006b8e28758f1a3107ae3 -O 703810
else
	# Tuning script is installed, or we are resuming. Start tuning script:
	export PATH=$HOME/miniconda/bin:$PATH
	export PATH=$HOME/cutechess/projects/cli:$PATH
	. "$HOME"/miniconda/etc/profile.d/conda.sh
	conda activate tuning
	cd "$HOME"/tuning || exit
	tune local -c config.json -v
fi
