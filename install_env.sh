#!/bin/bash
# 
# Installer for package
# 
# Run: ./install.sh
# 
# M. Ravasi, 24/05/2022

echo 'Creating my_env environment'

# create conda env
conda env create -f environment.yml
source ~/miniconda3/etc/profile.d/conda.sh
conda activate my_env
conda env list
echo 'Created and activated environment:' $(which python)

echo 'Done!'

