#!/bin/bash

#SBATCH --job-name=master-thesis-dik
#SBATCH --output=my_job_output_%j.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=16:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1


# Load modules or software if needed
module load python3
module load pytorch

# Execute the script or command
python src/main.py