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

# ab_remaining_pdbs_df = pd.read_csv("Extended_Abs_benchmark_remaining.csv")
# ab_remaining_pdbs_df =ab_remaining_pdbs_df.drop(columns=ab_remaining_pdbs_df.columns[:2])
# nb_remaining_pdbs_df = pd.read_csv("Nb_remaining_igfold_benchmark.csv")
# nb_remaining_pdbs_df = nb_remaining_pdbs_df.drop(columns=nb_remaining_pdbs_df.columns[:1])
# extended_bchmk_set = pd.concat([ab_remaining_pdbs_df,pd.read_csv("Nbs_final_seqs_withBUB.csv").drop(columns=['Unnamed: 0'])],axis=0,keys=['Ab','Nb']).reset_index().drop(columns=['level_1']).rename(columns={'level_0':'Protein_type'})

# AF2_abs_igfold_benchmark_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF2M_Benchmark/IgFold_remake/renamed_outputs/'
# AF2_nbs_igfold_benchmark_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF2M_Benchmark/IgFold_remake/Nbs/'
# Igfold_af3_preds_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/Ruffolo_AF3/'
# Native_clean_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/'
# AF3_pt1_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF3/sorted/'
# AF3_pt2_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/AF3/extended/'
# af3_igfold_dir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/Ruffolo_AF3/'
# bub_prot_to_AF3_extDir = {('bound','Nb'):"Bound/Nb/",('unbound','Nb'):"Unbound/Nb/",\
#                           ('bound','Ab'):"Bound/Ab/",('unbound','Ab'):"Unbound/Ab/"}
# bub_prot_to_Native_extDir = {('bound','Nb'):"bound_nanobody_pdbs/renamed/reordered/",('unbound','Nb'):"unbound_nanobody_pdbs/renamed/reordered/",\
#                           ('bound','Ab'):"bound_pdbs/renamed/reordered/",('unbound','Ab'):"unbound_pdbs/renamed/reordered/"}

# bub_prot_to_igfold_af3s_dir = {'Nb':'Abs/','Ab':'Nbs/'}


# datastructure = pd.read_csv("fixed_all_Ab_comparison_datastruc.csv").drop(columns=['Unnamed: 0'])
# nb_datastruc = pd.read_csv("fixed_all_Nb_comparison_datastruc.csv")
# datastructure = pd.concat([datastructure,nb_datastruc],keys=['antibody','nanobody']).reset_index().drop(columns=['level_1']).rename(columns={"level_0":"Protein_type"})


af3_v_native_data = pd.read_csv("af3_native_comparison_renumbered_datastruc_filtered_pdbs.csv").drop(columns=['Unnamed: 0']) # Newest
datastructure  = af3_v_native_data

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
    af3_file = datastructure.iloc[x].Dir_Pred+datastructure.iloc[x].Subdir+'/renamed_'+"_".join(af3_v_native_data.iloc[0].PDB_Pred.split("_")[-6:])
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
AF3_H3_plddts.to_csv("AF3_renum_all_h3_l3plddts.csv")
# AF2_h3_loops_rmsd_df = pd.concat([AF2_h3_loops_rmsd_df,pd.concat([pd.DataFrame(x,index=[0]) for x in results]).reset_index().drop(columns=['index'])],axis=1)
# AF2_h3_loops_rmsd_df
# # antigen_context_Ab_change_df