#!/bin/bash
eval "$(conda shell.bash hook)"

conda install mamba -n base -c conda-forge
mamba env create --file enviroment.yml

mamba run -n smart-tree pip install -e .

git clone --recursive https://github.com/lxxue/FRNN.git
mamba run -n smart-tree pip install -e FRNN/external/prefix_sum/.
mamba run -n smart-tree pip install -e FRNN/.