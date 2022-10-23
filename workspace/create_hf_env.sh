#!/bin/bash

conda create -y -n hf_env python=3.8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate hf_env
export PYTHONNOUSERSITE=True
pip install torch transformers
pip install conda-pack
conda-pack