#!/bin/bash
#SBATCH --job-name=model_search
#SBATCH --time=04:00:00
#SBATCH --mem=16GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1