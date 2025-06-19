#!/bin/bash
#SBATCH --job-name=IT
#SBATCH --nodes=1
#SBATCH --partition=ml-dept-p4de
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=10G
#SBATCH --time=128:00:00

python create_WB_wind.py

