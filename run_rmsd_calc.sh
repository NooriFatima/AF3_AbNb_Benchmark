#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH --account=jgray21
#SBATCH --time=4:30:00
#SBATCH -o out/Pyr_RMSD_calc_%j #send stdout to outfile
#SBATCH -e error/Pyr_RMSD_calc_%j #send stderr to errfile
#SBATCH --open-mode=append

which python3
conda activate AlphaFlow
python3 extended_pyrosetta_benchmarker.py
