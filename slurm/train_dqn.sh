#!/bin/bash
#SBATCH --job-name=train_dqn
#SBATCH --output=/home/mbassler/slurm_logs/pong_rl/%x_%j.log
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:30:00


echo "---START---"
date; pwd; hostname;
echo "$TMPDIR"

#load modules
echo "---LOADING MODULES---"
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
module load cuDNN/9.5.0.50-CUDA-12.6.0

#install packages
echo "---INSTALLING PACKAGES---"
pip install --user gymnasium[all] stable-baselines[extra]
pip install --user -e "$HOME"/github/pong_rl

#Run very simple script
echo "---RUNNING PYTHON SCRIPT---"
python "$HOME"/github/pong_rl/scripts/train.py \
 --dev=gpu \
 --logdir="$HOME"/output/pong_rl/tb_logs \
 --savedir="$HOME"/output/pong_rl

#log end
echo "---COMPLETED---"