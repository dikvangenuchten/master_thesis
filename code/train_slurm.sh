#!/bin/bash

#SBATCH --job-name=master-thesis-dik
#SBATCH --output=results/masther-thesis_%j.txt
#SBATCH --partition=mcs.gpu.q
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --gpus=1
#SBATCH --gres=gpu:1


# Load modules or software if needed
start=`date +%s`
module purge
module load Python/3.10.13-GCCcore-11.3.0
python -m venv venv
python -m pip install -r requirements_d.txt
python -m pip install -r requirements.txt
end=`date +%s`
echo Installing pacakges took: `expr $end - $start` seconds.

# Execute the script or command
echo "Starting Python script"
set -o allexport
source .env
set +o allexport
python src/hydra_main.py paths.datasets=/home/mcs001/20182591/master_thesis/code/ +mod=tue "$@"
