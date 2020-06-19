# Downloads and installs Lc0 v25, Stockfish 11, CuteChess
# as well as nets, openings, and syzygy TB files

mkdir tuning
mkdir syzygy

# Lc0
apt-get install clang-9 g++-8 nano ninja-build
pip install meson==0.52.1
git clone https://github.com/LeelaChessZero/lc0.git
cd lc0
git checkout release/0.25
CC=clang-9 CXX=clang++-9 ./build.sh
cp build/release/lc0 ../tuning/

cd ../

# Lc0 PR version
### ??? ###
# PR = "1288"
# git clone --recurse-submodules https://github.com/LeelaChessZero/lc0.git
# cd lc0
# git fetch origin pull/{PR}/head:PR{PR}
# git checkout {PR}
# git submodule update --init --remote
# rm -rf build
# mkdir -p build
# meson build --buildtype release -Dblas=false -Dopencl=false -Dcudnn=true -Dgtest=false
# build/ninja
# cp build/lc0 lc0_PR
### ??? ###

# Stockfish 11
cd tuning
wget https://stockfishchess.org/files/stockfish-11-linux.zip
unzip stockfish-11-linux.zip
cp stockfish-11-linux/Linux/stockfish_20011801_x64_bmi2 .
chmod 755 stockfish_20011801_x64_bmi2

cd ../

# CuteChess-CLI
apt-get install qt5-qmake qt5-default libqt5svg5-dev
git clone https://github.com/cutechess/cutechess.git
cd cutechess
qmake
make
cp projects/cli/cutechess-cli ../tuning/

cd ../

# Syzygy Endgame Tablebases
cd syzygy
wget -e robots=off -r -np http://tablebase.sesse.net/syzygy/3-4-5/
mv tablebase.sesse.net/syzygy/3-4-5/* .
md5sum -c checksum.md5

cd ../

# Oening Suites
cd tuning
wget https://cdn.discordapp.com/attachments/429710776282906625/536596158018224139/openings.zip
unzip openings.zip

# Net(s): 591226
wget https://training.lczero.org/get_network?sha=47e3f899519dc1bc95496a457b77730fce7b0b89b6187af5c01ecbbd02e88398 -O 591226

### TODO ###
# Create/upload engines.json
# Create/upload tuning.ipynb
