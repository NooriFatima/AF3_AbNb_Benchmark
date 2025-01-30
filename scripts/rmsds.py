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
# from Benchmarking.benchmark.ops.protein import *
# from Benchmarking.benchmark.ops.benchmark_clean_funcs import *
import colorcet as cc
import argparse
from pyrosetta import *
def exists(x):
    return x is not None
def init_pyrosetta(init_string=None, silent=True):
    if not exists(init_string):
        init_string = "-mute all -ignore_zero_occupancy false -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string, silent=silent)
init_pyrosetta()


parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()

igfold_datastruc = pd.read_csv(f"{args.datafilepath}")

rmsds= []
h3_loop_rmsds = []
af3_pdbs = []
af3_dirs = []
native_pdbs = []
native_dirs = []
bubs = []
ptypes = []
model_ = []
for i in trange(igfold_datastruc.shape[0]):
    af2 = igfold_datastruc.iloc[i].Dir_Pred+'/'+igfold_datastruc.iloc[i].PDB_Pred
    native = igfold_datastruc.iloc[i].Dir_Native + igfold_datastruc.iloc[i].PDB_Native
    try:
        af3_pose = pose_from_pdb(af2)
        native_pose = pose_from_pdb(native)
        results = get_ab_metrics(native_pose,af3_pose)
        print(results)
        rmsds.append(results)
        h3_loop_rmsd = scratch_CDRH3_RMSD(native_pose, af3_pose)
        print(f"\nloop rmsd: {h3_loop_rmsd}")
        h3_loop_rmsds.append(h3_loop_rmsd)
        af3_pdbs.append(igfold_datastruc.iloc[i].PDB_Pred)
        af3_dirs.append(igfold_datastruc.iloc[i].Dir_Pred)
        native_pdbs.append(igfold_datastruc.iloc[i].PDB_Native)
        native_dirs.append(igfold_datastruc.iloc[i].Dir_Native)
        bubs.append('unbound')
        ptypes.append(igfold_datastruc.iloc[i].Protein_type)
        # model_.append(igfold_datastruc.iloc[i].Model)
    except Exception as e:
        print(f"pdb {af2} errored out")
        print(e)

results_df = pd.concat([pd.DataFrame(rmsds[x],index=[x]) for x in range(len(rmsds))],axis=0).reset_index().drop(columns=['index'])
af3_rmsd_calcs = pd.concat([pd.DataFrame({"Pred_Dir":af3_dirs,"Pred_PDB":af3_pdbs,\
                                          "Bound_Unbound":bubs,'Protein_type':ptypes,\
                                            "Native_Dir":native_dirs,"Native_PDB":native_pdbs,\
                                                "H3_local_RMSD":h3_loop_rmsds}),results_df],axis=1)
af3_rmsd_calcs.to_csv(f"{args.resultsfilepath}")