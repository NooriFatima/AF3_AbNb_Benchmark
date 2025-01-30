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

AF2_abs_igfold_benchmark_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF2M_Benchmark/IgFold_remake/renamed_outputs/'
AF2_nbs_igfold_benchmark_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF2M_Benchmark/IgFold_remake/Nbs/'
Igfold_af3_preds_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/Ruffolo_AF3/'
Native_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/'
AF3_pt1_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF3/sorted/'
AF3_pt2_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF3/extended/'
af3_igfold_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/Ruffolo_AF3/'

bub_prot_to_AF3_extDir = {('bound','Nb'):"Bound/Nb/",('unbound','Nb'):"Unbound/Nb/",\
                          ('bound','Ab'):"Bound/Ab/",('unbound','Ab'):"Unbound/Ab/"}
bub_prot_to_Native_extDir = {('bound','Nb'):"bound_nanobody_pdbs/renamed/reordered/",('unbound','Nb'):"unbound_nanobody_pdbs/renamed/reordered/",\
                          ('bound','Ab'):"bound_pdbs/renamed/reordered/",('unbound','Ab'):"unbound_pdbs/renamed/reordered/"}

bub_prot_to_igfold_af3s_dir = {'Nb':'Abs/','Ab':'Nbs/'}

# ab_remaining_pdbs_df = pd.read_csv("Extended_Abs_benchmark_remaining.csv")
# ab_remaining_pdbs_df = ab_remaining_pdbs_df.drop(columns=ab_remaining_pdbs_df.columns[:2])
nb_remaining_pdbs_df = pd.read_csv("Nbs_final_seqs_withBUB.csv")
nb_remaining_pdbs_df = nb_remaining_pdbs_df.drop(columns=nb_remaining_pdbs_df.columns[:1])


# af3_ext_paired_results = []
# af3_ext_h3_loops_results = []
# # af3_nano_results = []
# af3_ext_pdbs = []
# for i in trange(ab_remaining_pdbs_df.shape[0]):
#     pdb_ = ab_remaining_pdbs_df.iloc[i].PDB.split('.')[0]
#     pdb_bub = ab_remaining_pdbs_df.iloc[i].Bound_Unbound
#     native_path = Native_clean_dir+bub_prot_to_Native_extDir[(pdb_bub,'Ab')]+pdb_+'.pdb'
#     if os.path.isfile(native_path) == False:
#         print(f'this native file does not exist: {native_path}')
#     else:
#         native_ = pose_from_pdb(Native_clean_dir+bub_prot_to_Native_extDir[(pdb_bub,'Ab')]+pdb_+'.pdb')
#         try:
#             # print(pdb_)
#             for k in range(1,4):
#                 if (os.path.isdir(AF3_pt2_dir+bub_prot_to_AF3_extDir[((pdb_bub,'Ab'))]+"fold_"+pdb_+f'_seed{k}/') == False):
#                     print('This af3 file does not exist: {}'.format(AF3_pt2_dir+bub_prot_to_AF3_extDir[((pdb_bub,'Ab'))]+"fold_"+pdb_+f'_seed{k}/'))
#                 else:
#                     for j in range(5):
#                         af3_pred = pose_from_pdb(AF3_pt2_dir+bub_prot_to_AF3_extDir[((pdb_bub,'Ab'))]+"fold_"+pdb_+f'_seed{k}/'+'renamed_fold_'+pdb_+f'_seed{k}_model_{j}'+'.pdb')
#                         af3_results = get_ab_metrics(native_,af3_pred)
#                         # print(af3_results)
#                         af3_h3_loop_rmsd = scratch_CDRH3_RMSD(native_, af3_pred)
#                         af3_ext_h3_loops_results.append(af3_h3_loop_rmsd)
#                         af3_ext_paired_results.append(af3_results)
#                         af3_ext_pdbs.append('renamed_fold_'+pdb_+f'_seed{k}_model_{j}')
#             # af2_pred = pose_from_pdb(AF2_abs_igfold_benchmark_clean_dir+pdb_+'.pdb')
#             # af2_results = get_ab_metrics(native_,af2_pred)
#             # af2_h3_loop_rmsd = scratch_CDRH3_RMSD(native_, af2_pred)
#             # af2_h3_loops_results.append(af2_h3_loop_rmsd)
#             # af2_paired_results.append(af2_results)
#             # af2_pdbs.append(pdb_)
#         except Exception as e:
#             print(e)

# Ab_AF3_extended_rosetta_benchmark_df = pd.concat([pd.DataFrame(x,index=['0']) for x in af3_ext_paired_results],axis=0).reset_index().drop(columns=['index'])
# Ab_AF3_extended_benchmark_df = pd.concat([pd.DataFrame({"PDB":af3_ext_pdbs,"h3_loop_rms":af3_ext_h3_loops_results}),Ab_AF3_extended_rosetta_benchmark_df],axis=1)
# Ab_AF3_extended_benchmark_df.to_csv('results/Ab_extendedset_pyrosetta_RMSD.csv')

af3_ext_nano_results = []
af3_ext_h3_nano_loops_results = []
# af3_nano_results = []
af3_ext_nano_pdbs = []
for i in trange(nb_remaining_pdbs_df.shape[0]):
    pdb_ = nb_remaining_pdbs_df.iloc[i].PDB.split('.')[0]
    pdb_bub = nb_remaining_pdbs_df.iloc[i].Bound_Unbound
    native_path = Native_clean_dir+bub_prot_to_Native_extDir[(pdb_bub,'Nb')]+pdb_+'.pdb'
    if os.path.isfile(native_path) == False:
        print(f'this native file does not exist: {native_path}')
    else:
        native_ = pose_from_pdb(Native_clean_dir+bub_prot_to_Native_extDir[(pdb_bub,'Nb')]+pdb_+'.pdb')
        try:
            # print(pdb_)
            for k in range(1,4):
                if (os.path.isdir(AF3_pt2_dir+bub_prot_to_AF3_extDir[((pdb_bub,'Nb'))]+"fold_"+pdb_+f'_seed{k}/') == False):
                    print('This af3 file does not exist: {}'.format(AF3_pt2_dir+bub_prot_to_AF3_extDir[((pdb_bub,'Nb'))]+"fold_"+pdb_+f'_seed{k}/'))
                else:
                    for j in range(5):
                        af3_pred = pose_from_pdb(AF3_pt2_dir+bub_prot_to_AF3_extDir[((pdb_bub,'Nb'))]+"fold_"+pdb_+f'_seed{k}/'+'renamed_fold_'+pdb_+f'_seed{k}_model_{j}'+'.pdb')
                        af3_results = get_ab_metrics(native_,af3_pred)
                        # print(af3_results)
                        af3_h3_loop_rmsd = scratch_CDRH3_RMSD(native_, af3_pred)
                        af3_ext_h3_nano_loops_results.append(af3_h3_loop_rmsd)
                        af3_ext_nano_results.append(af3_results)
                        af3_ext_nano_pdbs.append('renamed_fold_'+pdb_+f'_seed{k}_model_{j}')
            # af2_pred = pose_from_pdb(AF2_abs_igfold_benchmark_clean_dir+pdb_+'.pdb')
            # af2_results = get_ab_metrics(native_,af2_pred)
            # af2_h3_loop_rmsd = scratch_CDRH3_RMSD(native_, af2_pred)
            # af2_h3_loops_results.append(af2_h3_loop_rmsd)
            # af2_paired_results.append(af2_results)
            # af2_pdbs.append(pdb_)
        except Exception as e:
            print(e)

Nb_AF3_extended_rosetta_benchmark_df = pd.concat([pd.DataFrame(x,index=['0']) for x in af3_ext_nano_results],axis=0).reset_index().drop(columns=['index'])
Nb_AF3_extended_benchmark_df = pd.concat([pd.DataFrame({"PDB":af3_ext_nano_pdbs,"h3_loop_rms":af3_ext_h3_nano_loops_results}),Nb_AF3_extended_rosetta_benchmark_df],axis=1)
Nb_AF3_extended_benchmark_df.to_csv('results/Nb_extendedset_pyrosetta_RMSD.csv')