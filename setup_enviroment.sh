#!/bin/bash
python3 -m venv venv

# linux and macOS
source venv/bin/activate

# Windows
# .\venv\Scripts\activate

pip install emoji==0.6.0 
pip install packaging==20.9
pip install sparsemax
pip install numpy
pip install pandas
pip install transformers
pip install matplotlib
pip install flashtext
pip install torch
pip install pyspark
pip install contractions
pip install wordninja
pip install clean-text
pip install nltk
pip install torchvision

pip install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-2.0-cp38-cp38-linux_x86_64.whl
