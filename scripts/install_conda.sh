#!/usr/bin/env bash

export CONDA_ENV_NAME=vpare-env
echo $CONDA_ENV_NAME

conda create -n $CONDA_ENV_NAME python=3.7

eval "$(conda shell.bash hook)"
conda activate $CONDA_ENV_NAME
export ENV_PYTHON=$(conda info --base)/envs/$CONDA_ENV_NAME/bin/python

conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch

alias python=$ENV_PYTHON
which python
which pip

pip install pytorch-lightning==1.1.8 numpy==1.21.2
pip install git+https://github.com/giacaglia/pytube.git --upgrade
pip install -r requirements.txt
