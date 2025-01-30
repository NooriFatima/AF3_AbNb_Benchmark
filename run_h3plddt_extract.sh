#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH --account=<your partition account here>
#SBATCH --time=3:30:00
#SBATCH -o out/H3_plddt_%j #send stdout to outfile
#SBATCH -e error/H3_plddt_%j #send stderr to errfile
#SBATCH --open-mode=append

which python3
conda activate AF3
python3 scripts/h3_plddt_extract.py
