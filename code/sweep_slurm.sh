#!/bin/bash

echo Preloading packages
start=`date +%s`
module purge
module load Python/3.10.13-GCCcore-11.3.0
python -m venv venv
python -m pip install -qr requirements_d.txt
python -m pip install -qr requirements.txt
end=`date +%s`
echo Installing pacakges took: `expr $end - $start` seconds.

# Starting sweep
echo "Exporting env variables"
set -o allexport
source .env
set +o allexport
cd src
echo "Starting Python script"
python hydra_main.py -m +mod=tue hydra/launcher=tue_slurm "$@"