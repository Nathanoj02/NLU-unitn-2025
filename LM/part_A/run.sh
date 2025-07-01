#!/bin/bash

#SBATCH --job-name=nlu_lm_a
#SBATCH --output=nlu_lm_a%j.out
#SBATCH --error=nlu_lm_a%j.err
#SBATCH --partition=edu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1

python3 main.py