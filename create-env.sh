#!/bin/bash
eval "$(conda shell.bash hook)"

conda create -n smart-tree python=3.10
conda run -n smart-tree conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda run -n smart-tree pip install -e .

echo Installing FRNN
git clone --recursive https://github.com/lxxue/FRNN.git
conda run -n smart-tree pip install -e FRNN/external/prefix_sum/.
conda run -n smart-tree pip install -e FRNN/.

conda run -n smart-tree conda install cudf=24.02 cugraph=24.02 -c rapidsai -c conda-forge -c nvidia
