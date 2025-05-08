#!/bin/bash
#SBATCH --job-name=diffuse_run
#SBATCH --partition=normal-a100-80                                   
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --output=run.out
#SBATCH --error="err.txt"
#SBATCH --account=f202500002hpcvlabistulg

source /projects/F202500002HPCVLABISTUL/inescgarcia/Dif-fuse/diffuse_env/bin/activate

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Train DDPM with healthy images
srun --ntasks=4 python scripts/image_train.py
