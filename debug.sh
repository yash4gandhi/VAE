#!/bin/bash

#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=p100:1
#SBATCH --time=1:00:00

module purge
eval "$(conda shell.bash hook)"
conda activate vae

python vae_script.py