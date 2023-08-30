#!/bin/bash
eval "$(conda shell.bash hook)"

conda env create --file enviroment.yml

git clone --recursive https://github.com/lxxue/FRNN.git
conda run -n smart-tree pip install -e FRNN/external/prefix_sum/.
conda run -n smart-tree pip install -e FRNN/.