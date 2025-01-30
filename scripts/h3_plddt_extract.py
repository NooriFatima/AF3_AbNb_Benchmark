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
import argparse

from Benchmarking.benchmark.ops.all_funcs import *
from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.benchmark_clean_funcs import *

parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()

datastructure = pd.read_csv(f"{args.datafilepath}").drop(columns=['Unnamed: 0']) # Newest

#Getting plddts for extended set


AF3pdbs = []
af3_prottype=[]
af3_bub = []
h3_plddts = []
h3_plddt_arrays =[]
l3_plddts = []
l3_plddt_arrays =[]
af3_pt = []
for x in trange(datastructure.shape[0]):
    # native_loc = native_base_dir+prot_bub_tonative_dict[(AF2_df.iloc[i].Bound_Unbound,AF2_df.iloc[i].Protein_type)]+AF2_df.iloc[i].PDB
    # af2_loc = af2_base_dir+AF2_df.iloc[i].Protein_type+"/"+AF2_df.iloc[i].PDB_Full
    # af3_outer_dir = AF3_pt2_dir+bub_prot_to_AF3_extDir[(extended_bchmk_set.iloc[i].Bound_Unbound,extended_bchmk_set.iloc[i].Protein_type)]
    # for j in range(1,4):
    #     sub_dir =f"fold_{extended_bchmk_set.iloc[i].PDB.split('.')[0]}_seed{j}/"
    #     if os.path.isdir(af3_outer_dir+sub_dir):
    #         per_pdbs = [x for x in os.listdir(af3_outer_dir+sub_dir) if 'renamed' in x]
    #         for k in per_pdbs:
    af3_file = datastructure.iloc[x].Dir_Pred+datastructure.iloc[x].Subdir+'/renamed_'+"_".join(datastructure.iloc[0].PDB_Pred.split("_")[-6:])
    # print(af3_loc)
    if os.path.isfile(af3_file):
        try:
            # native_pose = pose_from_pdb(native_loc)
            af3_pose = pose_from_pdb(af3_file)
            h3_plddt_array = CDRH3_Bfactors(af3_pose)
            l3_plddt_array = CDRL3_Bfactors(af3_pose)
            # print(h3_plddt_array)
            h3_plddt_arrays.append(h3_plddt_array)
            l3_plddt_arrays.append(l3_plddt_array)
            # af3_prottype.append(datastructure.iloc[x].Protein_type)
            af3_bub.append(datastructure.iloc[x].Bound_Unbound)
            af3_pt.append(datastructure.iloc[x].Protein_type)
            # nativepdbs.append(AF2_df.iloc[i].PDB)
            AF3pdbs.append(datastructure.iloc[x].PDB_Pred)
            # h3_loop_rmsds.append(h3_loop_rmsd)
            # print(h3_loop_rmsd)
            print(np.array(h3_plddt_array).mean())
        except Exception as e:
            print(e)
    else:
        pass
AF3_H3_plddts = pd.DataFrame({"PDB":AF3pdbs,'H3_plddt_arrays':h3_plddt_arrays,'L3_plddt_arrays':l3_plddt_arrays,\
                              'Avg_H3_pLDDT':[np.array(x).mean() for x in h3_plddt_arrays],\
                              'Avg_L3_pLDDT':[np.array(x).mean() for x in l3_plddt_arrays],\
                              "Bound_Unbound":af3_bub})
AF3_H3_plddts.to_csv(f"{args.resultsfilepath}")
