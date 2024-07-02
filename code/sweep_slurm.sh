#!/bin/bash

start=`date +%s`
module purge
module load Python/3.10.13-GCCcore-11.3.0
python -m venv venv
python -m pip install -qr requirements_d.txt
python -m pip install -qr requirements.txt
end=`date +%s`
echo Installing pacakges took: `expr $end - $start` seconds.

# Starting sweep
echo "Starting Python script"
set -o allexport
source .env
set +o allexport
python src/hydra_main.py -m +mod=tue hydra/launcher=tue_slurm "$@"