#!/bin/bash
eval "$(conda shell.bash hook)"

# echo "before calling source: $PATH"

conda install mamba -n base -c conda-forge
mamba env create --file enviroment.yml
conda activate smart-tree

pip install --upgrade pip setuptools

# echo "before calling source: $PATH"

git clone --recursive https://github.com/lxxue/FRNN.git
cd FRNN/external/prefix_sum
pip install .
cd ../../ # back to the {FRNN} directory
pip install .
cd ..

pip install .





