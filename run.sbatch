#!/bin/bash

#SBATCH --job-name=LLM_example
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --time=01-12:00:00
#SBATCH --cpus-per-gpu=32
#SBATCH --mem-per-gpu=32G
#SBATCH --qos=normal

# Source bashrc and explicitly initialize conda
source $HOME/.bashrc
#source $HOME/miniconda3/etc/profile.d/conda.sh  # adjust path if necessary

# Activate your conda environment
conda activate Cramming_example
#conda install matplotlib
#pip install tiktoken

# Debug: Check that Python is available
which python
python --version

nvidia-smi

echo "#############################################################"
echo "### starting Training with Rilke books on GPU ###"
echo "#############################################################"

python /home/cluster_home/simbeck/RilkeLM/GPTAndPrejudice/2_pretrain.py



########aus alter Batch#############
#known good single GPU:
#python cramming/pretrain.py name=test arch=bert-base train=bert-base data=sanity-check-2 impl.microbatch_size=2

#known good multi GPU
#torchrun --nproc_per_node=2 --standalone cramming/pretrain.py name=multi_GPU_2_cramming  arch=bert-base train=bert-base data=sanity-check-2
