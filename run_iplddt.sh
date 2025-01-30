#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --partition=shared
#SBATCH --account=jgray21
#SBATCH --time=24:00:00
#SBATCH -o out/iplddt_%j #send stdout to outfile
#SBATCH -e error/iplddt_%j #send stderr to errfile
#SBATCH --open-mode=append

which python3
conda activate AF3
python3 iplddt_calc.py
