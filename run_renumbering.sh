#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH --account=jgray21
#SBATCH --time=12:30:00
#SBATCH -o out/igfoldrenum_%j #send stdout to outfile
#SBATCH -e error/igfoldrenum_%j #send stderr to errfile
#SBATCH --open-mode=append

which python3
conda activate AF3
python3 renum_igfold_af3.py
