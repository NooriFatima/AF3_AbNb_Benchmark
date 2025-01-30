import requests
from biopandas.pdb import PandasPdb
import scipy as sp
import pandas as pd
import os
import numpy as np
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
from Benchmarking.benchmark.ops.all_funcs import * 
from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.benchmark_clean_funcs import *
import colorcet as cc



af3_igfold_renamed_datastruc = pd.read_csv("af3_igfold_benchmarkrenamed_datastruc_nbs.csv").drop(columns=['Unnamed: 0'])
af3_igfold_renamed_datastruc
for i in trange(af3_igfold_renamed_datastruc.shape[0]):
    pdb_src = af3_igfold_renamed_datastruc.iloc[i].Dir +'/'+ af3_igfold_renamed_datastruc.iloc[i].PDB
    pdb_dest = af3_igfold_renamed_datastruc.iloc[i].Dir + "/renum_"+af3_igfold_renamed_datastruc.iloc[i].PDB
    if os.path.isfile(pdb_dest):
        pass
    else:
        renumber_pdb(pdb_src,pdb_dest)