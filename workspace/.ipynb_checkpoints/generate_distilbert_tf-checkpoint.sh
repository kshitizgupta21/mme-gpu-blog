#!/bin/bash


# use this script to export DistilBERT Classifcation TF model as a savedmodel file

echo "Installing Transformers..."
pip -q install transformers

python tf_exporter.py
