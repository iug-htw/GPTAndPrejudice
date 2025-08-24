#!/bin/bash
#SBATCH --job-name=gpt_prej_saes
#SBATCH --output=%x-%A.out
#SBATCH --error=%x-%A.err
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-0:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem-per-gpu=40G

set -x
export PYTHONUNBUFFERED=1

source /etc/profile.d/conda.sh
conda activate /scratch/mahran/project/.conda-env

cd /scratch/mahran

TRAIN_FILE=${TRAIN_FILE:-train_text_data.txt}
VAL_FILE=${VAL_FILE:-val_text_data.txt}

EMB_OUT=${EMB_OUT:-sae_data}        # where .npy embeddings are written
SAE_OUT=${SAE_OUT:-sae_models}      # where SAE checkpoints and logs go
LAYERS="${LAYERS:-1 2 3 4 5 6 7 8}" # layers to extract + train on

mkdir -p "$EMB_OUT" "$SAE_OUT"

# echo "==== [Step 1/2] Extracting embeddings to $EMB_OUT ===="
# srun python extract_embeddings.py \
#   --train_file "$TRAIN_FILE" \
#   --val_file "$VAL_FILE" \
#   --out_dir "$EMB_OUT" \
#   --ckpt "model_896_14_8_256.pth" \
#   || { echo "Extract embeddings FAILED"; exit 1; }

# echo "==== [Step 2/2] Training SAEs per layer; saving to $SAE_OUT ===="
echo "==== Training SAEs per layer; saving to $SAE_OUT ===="
srun python train_saes.py \
  --data_dir "$EMB_OUT" \
  --top_k 50 \
  --epochs 500 \
  --batch_size 64 \
  --lr 5e-4 \
  --weight_decay 1e-6 \
  --patience 10 \
  --out_dir "$SAE_OUT" \
  --device "cuda" \
  || { echo "Train SAEs FAILED"; exit 1; }

# echo "✅ Done. Embeddings in '$EMB_OUT', SAE checkpoints in '$SAE_OUT'."
echo "✅ Done. SAE checkpoints in '$SAE_OUT'."