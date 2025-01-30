#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH --account=jgray21
#SBATCH --time=24:00:00
#SBATCH -o out/all_energies_%j #send stdout to outfile
#SBATCH -e error/all_energies_%j #send stderr to errfile
#SBATCH --open-mode=append

which python3
conda activate AF3
python3 interface_fast_relaxer.py
