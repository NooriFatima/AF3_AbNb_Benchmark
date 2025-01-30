from biopandas.pdb import PandasPdb
import pandas as pd
import os
import numpy as np
import shutil
import Bio
from abnumber import Chain
import math
import enum
import torch
import torch.nn.functional as F
from tqdm import trange
from Bio.PDB import PDBParser
from Bio.PDB.Selection import unfold_entities
from Bio.SeqIO import PdbIO
import warnings
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Sequence, List, Optional, Union
from Levenshtein import distance, ratio
from Bio.Align.Applications import ClustalwCommandline
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
import argparse
from Benchmarking.benchmark.ops.all_funcs import *
from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.benchmark_clean_funcs import *

parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()

datastructure = pd.read_csv(f"{args.datafilepath}")

for i in trange(datastructure.shape[0]):
    pdb_row = datastructure.iloc[i]
    filepath = pdb_row.Dir + pdb_row.PDB
    dest_filepath = pdb_row.Dir + 'PDBs/'+pdb_row.PDB
    convert_mmcif_pdb(filepath,dest_filepath)

