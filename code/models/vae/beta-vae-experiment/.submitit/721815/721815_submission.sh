#!/bin/bash

# Parameters
#SBATCH --array=0-7%8
#SBATCH --cpus-per-gpu=7
#SBATCH --error=/home/mcs001/20182591/master_thesis/code/src/multirun/2024-07-05/13-43-47/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gpus-per-task=1
#SBATCH --job-name=thesis-dik-hydra_main
#SBATCH --mail-user=h.j.m.v.genuchten@student.tue.nl
#SBATCH --mem-per-gpu=32G
#SBATCH --nice=1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/home/mcs001/20182591/master_thesis/code/src/multirun/2024-07-05/13-43-47/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --partition=mcs.gpu.q
#SBATCH --signal=USR2@120
#SBATCH --time=600
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /home/mcs001/20182591/master_thesis/code/src/multirun/2024-07-05/13-43-47/.submitit/%A_%a/%A_%a_%t_log.out --error /home/mcs001/20182591/master_thesis/code/src/multirun/2024-07-05/13-43-47/.submitit/%A_%a/%A_%a_%t_log.err /sw/rl8/zen/app/Python/3.10.13-GCCcore-11.3.0/bin/python -u -m submitit.core._submit /home/mcs001/20182591/master_thesis/code/src/multirun/2024-07-05/13-43-47/.submitit/%j
