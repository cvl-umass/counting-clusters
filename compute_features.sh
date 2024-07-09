GPU=${1:-0}
DATASET="MacaqueFaces"
MODEL="L-384"

CUDA_VISIBLE_DEVICES=$GPU python3.6 src/compute_features.py --dataset $DATASET --save_dir features --cuda --model $MODEL
