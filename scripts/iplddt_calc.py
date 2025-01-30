from biopandas.pdb import PandasPdb
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
import argparse
from Bio.PDB import PDBIO
from Bio.PDB.Superimposer import Superimposer
import urllib.request


parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()


datastructure = pd.read_csv(f"{args.datafilepath}").drop(columns=['Unnamed: 0'])
# nb_datastruc = pd.read_csv("fixed_all_Nb_comparison_datastruc.csv")
# datastructure = pd.concat([datastructure,nb_datastruc],keys=['antibody','nanobody']).reset_index().drop(columns=['level_1']).rename(columns={"level_0":"Protein_type"})



iplddts = []
AF3_PDBs=[]
ptypes = []

for i in trange(datastructure.shape[0]):
# for i in trange(1):
    path_ = datastructure.iloc[i].Dir_Pred+datastructure.iloc[i].Subdir+'/renamed_'+"_".join(datastructure.iloc[i].PDB_Pred.split("_")[1:])
    protein = datastructure.iloc[i].Protein_type
    if protein == 'antibody':
        partners = 'HL_A'
    else:
        partners = 'H_A'
    interface_plddts,iplddt = get_interface_res_fast(path_,partners=partners)
    if interface_plddts!=None:
    # i_plddt = np.mean(list(interface_plddts.values()))
        print(iplddt)
        AF3_PDBs.append('renamed_'+"_".join(datastructure.iloc[i].PDB_Pred.split("_")[1:]))
        iplddts.append(iplddt)
        ptypes.append(protein)
    else:
        pass

iplddt_df = pd.DataFrame({"AF3_PDB":AF3_PDBs,"I_pLDDT":iplddts,'Protein_type':ptypes})
iplddt_df.to_csv(f"{args.resultsfilepath}")