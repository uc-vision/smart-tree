#!/bin/bash
eval "$(conda shell.bash hook)"

conda env create --file enviroment.yml &&
echo Enviroment Created... &&
git clone --recursive https://github.com/lxxue/FRNN.git 
conda run -n smart-tree pip install -e FRNN/external/prefix_sum/. &&
conda run -n smart-tree pip install -e FRNN/. &&
conda run -n smart-tree conda install -c  &&
echo Installed 
#conda run -n smart-tree install rapids=24.02 cuda-version=12.0 -c rapidsai -c conda-forge -c nvidia 