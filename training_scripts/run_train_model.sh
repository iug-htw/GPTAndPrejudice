#!/bin/bash
#SBATCH --job-name=gpt_prej
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=20G

source /etc/profile.d/conda.sh
conda activate /scratch/mahran/project/.conda-env

cd /scratch/mahran
python train_gpt_model.py

