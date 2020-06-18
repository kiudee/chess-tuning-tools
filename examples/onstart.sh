################################################
# vastai/tensorflow with Jupyter Lab interface #
################################################

# Update installers
pip install --upgrade pip
apt-get update

# Install Conda
# Ref: https://medium.com/@error.replicator/setting-up-cloud-environment-for-deep-learning-febb5c408e78
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
~/miniconda3/bin/conda init
source .bashrc

# Create new python kernel
conda create -n bask python=3.7 ipykernel -y
conda activate bask

# For issues with psycopg2: stackoverflow.com/questions/35104097
apt install -y libpq-dev postgresql-server-dev-all

# Install chess-tuning-tools
pip install chess-tuning-tools

# Install "bask" kernel
python -m ipykernel install --user --name=bask
