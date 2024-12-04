#!/bin/bash
# Activate the conda environment
#source ~/miniconda3/etc/profile.d/conda.sh  # Adjust the path if necessary
#conda activate evodenss

dataset_name="argo"
config_path="config_files/argo.yaml"
grammar_path="grammars/argo.grammar"


python -m evodenss.main \
    -d "$dataset_name" \
    -c "$config_path" \
    -g "$grammar_path" \
    -r 16\
    --gpu-enabled

    

#conda deactivate
