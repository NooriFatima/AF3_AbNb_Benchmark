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
import argparse
from Benchmarking.benchmark.ops.all_funcs import *
from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.benchmark_clean_funcs import *
from pyrosetta.rosetta.core.scoring import *
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from Bio import BiopythonWarning

warnings.simplefilter('ignore', BiopythonWarning)
init('-ignore_unrecognized_res \
     -ignore_zero_occupancy false -load_PDB_components false \
     -no_fconfig -check_cdr_chainbreaks false')

parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()


def get_interface_analyzer(partner_chain_str, scorefxn, pack_sep=False):
  interface_analyzer = pyrosetta.rosetta.protocols.analysis.InterfaceAnalyzerMover()
  interface_analyzer.fresh_instance()
  interface_analyzer.set_pack_input(True)
  interface_analyzer.set_interface(partner_chain_str)
  interface_analyzer.set_scorefunction(scorefxn)
  interface_analyzer.set_compute_interface_energy(True)
  interface_analyzer.set_compute_interface_sc(True)
  interface_analyzer.set_calc_dSASA(True)
  interface_analyzer.set_pack_separated(pack_sep)

  return interface_analyzer

def interface_energy_calc(pred_pdb,pack_scorefxn,prot_type):
    if prot_type == 'antibody':
        pred_pdb_Ag = [x for x in list(get_atmseq(pred_pdb)) if x not in ['H','L']]
        pred_pose = pose_from_pdb(pred_pdb)
        pred_interface = f"HL_{pred_pdb_Ag}"
    elif prot_type =='nanobody':
        pred_pdb_Ag = [x for x in list(get_atmseq(pred_pdb)) if x not in ['H']]
        pred_pose = pose_from_pdb(pred_pdb)
        pred_interface = f"H_{pred_pdb_Ag}"
    interface_analyzer = get_interface_analyzer(pred_interface, pack_scorefxn)
    interface_analyzer.apply(pred_pose)
    interface_analyzer_packsep = get_interface_analyzer(pred_interface, pack_scorefxn, pack_sep=True)
    interface_analyzer_packsep.apply(pred_pose)
    binding_energy_dgsep = interface_analyzer_packsep.get_interface_dG()
    return binding_energy_dgsep
   
   

pdbs = []
prottypes = []
bind_es = []
dirs = []

energy_fxn = "ref2015"
pack_scorefxn = create_score_function(energy_fxn)

# dirs = [AF3_og_ab_dir,AF3_ext_ab_dir]

# datastructure = pd.read_csv("results/AF3_results_renum_fv.csv").drop(columns=['Unnamed: 0'])
# datastructure['Protein_type'] = ['nanobody' if datastructure.iloc[x].ocd==0.0 else 'antibody' for x in range(datastructure.shape[0])]
# datastructure= datastructure[(datastructure['Bound_Unbound']=='bound')&(datastructure['AF3_PDB'].str.contains('seed1'))&(datastructure['AF3_PDB'].str.contains('model_0'))]
datastructure = pd.read_csv(f"{args.datafilepath}")
datastructure = datastructure[datastructure['Bound_Unbound']=='bound']

for i in trange(datastructure.shape[0]):
    path_ = datastructure.iloc[i].AF3_Dir +  datastructure.iloc[i].AF3_PDB
    prottype = datastructure.iloc[i].Protein_type
    # print(i+j+"/"+k)
    b_E = interface_energy_calc(path_,pack_scorefxn,prottype)
    bind_es.append(b_E)
    dirs.append(datastructure.iloc[i].AF3_Dir)
    pdbs.append(datastructure.iloc[i].AF3_PDB)
    prottypes.append(prottype)

binding_es = pd.DataFrame({"PDB":pdbs,"del_G_B":bind_es,'Protein_type':prottypes})
binding_es.to_csv(f"{args.resultsfilepath}")

# dirs =[AF3_og_nb_dir,AF3_ext_nb_dir]
# dirs =[AF3_ext_nb_dir]
# pdbs = []
# prottypes = []
# bind_es = []
# for i in dirs:
#     subdirs = [x for x in os.listdir(i) if '.zip' not in x and 'seed1' in x]
#     if 'Nb' in i:
#         prottype = 'nanobody'
#     else:
#         prottype = 'antibody'
#         print('prottype: {}'.format(prottype))
#     print(subdirs)
#     for j in trange(len(subdirs)):
#         print(j)
#         pred_pdbs = [x for x in os.listdir(i+subdirs[j]) if 'renamed_' in x]
#         print(pred_pdbs)
#         for k in pred_pdbs:
#             print(k)
#         #         print('prottype: {}'.format(prottype))
    
#             b_E = interface_energy_calc(i+subdirs[j]+"/"+k,pack_scorefxn,prottype)
#             bind_es.append(b_E)
#             pdbs.append(k)
#             prottypes.append(prottype)

# binding_es = pd.DataFrame({"PDB":pdbs,"del_G_B":bind_es,'Protein_type':prottypes})
# binding_es.to_csv("results/AF3_ext_Nb_Benchmark_bindingenergies.csv")