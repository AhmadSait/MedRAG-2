#!/bin/bash
#SBATCH --job-name=llama
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --constraint=a100
#SBATCH --time=24:00:00
module load cuda/11.8
source /ibex/user/zhuw0b/miniforge/bin/activate /ibex/user/zhuw0b/conda-environments/llama
python eval_llama3_8B.py "/ibex/user/zhuw0b/MIRAGE/benchmark.json"