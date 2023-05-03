#!/bin/bash

#SBATCH --partition=g40
#SBATCH --job-name=dataset2metadata
#SBATCH --output=logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --comment=datanet
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem 100G
#SBATCH --array=1-640%64
#SBATCH --requeue

echo "Processing job $SLURM_ARRAY_TASK_ID.yml"

cd /fsx/home-sy/scratch

FILE=logs/
if test -d "./$FILE"; then
    echo ""
else
    echo "$FILE does not exist in processing dir."
    exit 1
fi

export PATH="/fsx/home-$USER/miniconda3/condabin:$PATH"
source /fsx/home-$USER/miniconda3/etc/profile.d/conda.sh

conda activate tng_metadata
srun dataset2metadata --yml examples/jobs/$SLURM_ARRAY_TASK_ID.yml
