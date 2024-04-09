#!/bin/bash

#SBATCH --job-name=master-thesis-dik
#SBATCH --output=~/masther-thesis_%j.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1


# Load modules or software if needed
module purge
module load Python/3.10.13-GCCcore-11.3.0
python -m venv venv
python -m pip install -r requirements_d.txt
python -m pip install -r requirements.txt

# Execute the script or command
python src/main.py
