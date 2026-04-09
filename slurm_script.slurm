#!/bin/bash --login
#SBATCH -J trgsrmcl2
#SBATCH -p gpuA40GB
#SBATCH -G 1
#SBATCH -t 0-4

module purge
module load libs/cuda/11.7.0

python training.py
