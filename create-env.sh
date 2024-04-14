#!/bin/bash
eval "$(conda shell.bash hook)"

conda create -n smart-tree python=3.10 &&
conda run -n smart-tree conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit &&
conda run -n smart-tree conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia &&
conda run -n smart-tree conda install gxx_linux-64 && # For installing FRNN &&
conda run -n smart-tree pip install . &&

echo Installing FRNN
git clone --recursive https://github.com/lxxue/FRNN.git 
conda run -n smart-tree pip install FRNN/external/prefix_sum/. &&
conda run -n smart-tree pip install -e FRNN/. &&

#conda run -n smart-tree conda install -c rapidsai -c conda-forge -c nvidia cudf=23.12 cugraph=23.12  python=3.10 cuda-version=11.8 --solver=libmamba 
conda run -n smart-tree conda install -c rapidsai -c conda-forge -c nvidia cudf=23.02 cugraph=23.02 python=3.10 cuda-version=11.8
