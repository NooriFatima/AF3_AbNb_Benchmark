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


from DockQ.DockQ import load_PDB, run_on_all_native_interfaces

# datastructure = pd.read_csv("fixed_all_Ab_comparison_datastruc.csv").drop(columns=['Unnamed: 0'])
# nb_datastruc = pd.read_csv("fixed_all_Nb_comparison_datastruc.csv")
# datastructure = pd.concat([datastructure,nb_datastruc])
# datastructure = datastructure[datastructure['Bound_Unbound']=='bound']
# notrun_pdbs = ['8h7m_1.pdb','8h7i_0.pdb','8h7n_0.pdb','8h7r_0.pdb','7te8_2.pdb','8w85_0.pdb','8c7h_0.pdb','8w86_0.pdb','8w84_0.pdb']
# notrun_pdbs = ['8h7m_1','8h7i_0','8h7n_0','8h7r_0','7te8_2','8w85_0','8c7h_0','8w86_0','8w84_0']
# af3_v_native_data = pd.read_csv("af3_native_comparison_renumbered_datastruc_filtered_pdbs.csv").drop(columns=['Unnamed: 0'])
# datastructure = af3_v_native_data[af3_v_native_data['Bound_Unbound']=='bound']
# datastructure = datastructure[datastructure['PDB_short'].isin(notrun_pdbs)]
# pred_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/seed_div_followup/7wtf_1_pdbs/'
# pred_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/seed_div_followup/hierarchical_followup/'
# subdirs = [x for x in os.listdir(pred_dir) if ('.zip' not in x) & ('fab' not in x) & ('fvag' not in x)]
# native_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/bound_pdbs/renamed/reordered/'

datastructure = pd.read_csv("AlphaRED/preds/alphared_rmsd_datastruc.csv")

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
all_dock_scores.to_csv("AlphaRED/results/AlphaRED_all_dockqs.csv")