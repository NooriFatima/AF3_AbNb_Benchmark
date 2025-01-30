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


from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()

datastructure = pd.read_csv(f"{args.datafilepath}")

pdbs = []
dockqs = []
irmss = []
lrmss = []
f1s = []
fnats = []
seeds = []
models = []
ptypes = []
dirs = []
# for i in trange(2):
for i in trange(datastructure.shape[0]):
    # pred_pdbs = [x for x in os.listdir(pred_dir+i) if "renamed_" in x]
    # for pdb in pred_pdbs:
    pdb = datastructure.iloc[i].PDB #_Pred
    ptype = datastructure.iloc[i].Protein_type
    native_pdb = datastructure.iloc[i].PDB_Native  #"_".join(i.split("_")[1:3])+".pdb"
    native_dir = datastructure.iloc[i].Dir_Native
    try:
        # print(f"{native_dir}{native_pdb}")
        # print([x for x in list(get_atmseq(f"{native_dir}{native_pdb}").keys())])
        chain_map = {'H':'H',[x for x in list(get_atmseq(f"{native_dir}{native_pdb}").keys()) if x!='H' and x!='L'][0]:'A'}
        # print(chain_map)
        # pred_path = datastructure.iloc[i].Dir_Pred+datastructure.iloc[i].Subdir+"/"+pdb #pred_dir+i+"/"+pdb
        pred_path = datastructure.iloc[i].Dir+pdb
        # print(pred_path)
        model = load_PDB(f"{pred_path}")
        native = load_PDB(f"{native_dir}{native_pdb}")
        # print(native)
        Dockq_dict = run_on_all_native_interfaces(model, native, chain_map=chain_map)
        print('found the dockq dict')
        print(Dockq_dict)
        Dockq_score = Dockq_dict[0][('HA')]['DockQ']
        print(f"this is the dockq score: {Dockq_score}")
        F1_score = Dockq_dict[0][('HA')]['F1']
        print('got f1 score')
        irms_score = Dockq_dict[0][('HA')]['iRMSD']
        print('got irms score')
        lrms_score = Dockq_dict[0][('HA')]['LRMSD']
        print('got lrms score')
        fnat_score = Dockq_dict[0][('HA')]['fnat']
        print('got fnat score')
        dockqs.append(Dockq_score)
        fnats.append(fnat_score)
        lrmss.append(lrms_score)
        f1s.append(F1_score)
        irmss.append(irms_score)
        dirs.append(datastructure.iloc[i].Dir)
        print(pdb)
        seeds.append(pdb.split('_')[5].split('d')[1])
        models.append(pdb.split('_')[-1].split('.')[0])
        pdbs.append(pdb)
        ptypes.append(ptype)
        print("pdb {} has a dockq score of {}".format(pdb,Dockq_score))
    except Exception as e:
        print(e)

all_dock_scores = pd.DataFrame({'Dir':dirs,'PDB':pdbs,'Protein_type':ptypes,"DockQ":dockqs,"iRMS":irmss,"LRMS":lrmss,"F1":f1s,"Fnat":fnats,'Seed':seeds,'Model':models})
all_dock_scores.to_csv(f"{args.resultsfilepath}")