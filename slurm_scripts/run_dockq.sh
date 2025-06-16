#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH --account=<your partition account here>
#SBATCH --time=4:30:00
#SBATCH -o out/af3_dockq_%j #send stdout to outfile
#SBATCH -e error/af3_dockq_%j #send stderr to errfile
#SBATCH --open-mode=append

which python3
conda activate AF3
python3 scripts/dockq_calc_withdatastruc.py
