#!/bin/bash
#SBATCH --job-name=test_job
#SBATCH --output=output.log
#SBATCH --error=error.log
#SBATCH --time=01:00:00
#SBATCH --partition=it-hpc
#SBATCH --gres=gpu:1
#SBATCH --time=128:00:00

module purge

sleep infinity
