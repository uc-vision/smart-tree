FROM python:3.10

WORKDIR /usr/src/app

COPY . .

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

ENV PATH=$CONDA_DIR/bin:$PATH

RUN conda install -n base conda-libmamba-solver
RUN conda config --set solver libmamba

# Create conda environment
RUN bash create-env.sh

CMD run-smart-tree +path=test_data/output.ply pipeline.view_model_output=False pipeline.view_skeletons=False

ENV NAME smart_tree