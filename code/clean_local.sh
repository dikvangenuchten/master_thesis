#!/bin/bash

#SBATCH --job-name=master-thesis-dik
#SBATCH --output=results/masther-thesis_%j.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=0
#SBATCH --gres=gpu:1

rm -r /local/20182591/