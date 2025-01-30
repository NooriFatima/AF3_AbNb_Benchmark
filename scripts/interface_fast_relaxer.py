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
#Python
from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.teaching import *

#Core Includes
from rosetta.core.kinematics import MoveMap
from rosetta.core.kinematics import FoldTree
from rosetta.core.pack.task import TaskFactory
from rosetta.core.pack.task import operation
from rosetta.core.simple_metrics import metrics
from rosetta.core.select import residue_selector as selections
from rosetta.core import select
from rosetta.core.select.movemap import *

#Protocol Includes
from rosetta.protocols import minimization_packing as pack_min
from rosetta.protocols import relax as rel
from rosetta.protocols.antibody.residue_selector import CDRResidueSelector
from rosetta.protocols.antibody import *
from rosetta.protocols.loops import *
from rosetta.protocols.relax import FastRelax

def exists(x):
    return x is not None
def init_pyrosetta(init_string=None, silent=True):
    if not exists(init_string):
        init_string = '-use_input_sc -input_ab_scheme Chothia -ignore_unrecognized_res \
     -ignore_zero_occupancy false -load_PDB_components false -relax:default_repeats 2 -no_fconfig' # -ex1 -ex2'
        # init_string = "-mute all -ignore_zero_occupancy false -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string, silent=silent)
init_pyrosetta()

# run overall relax + regional (interface only) relax

# final_af3_benchmark = pd.read_csv("results/round_6/datafiles/af3_withauxdata_results.csv").drop(columns=['Unnamed: 0'])
# fulldelgb_df = pd.read_csv("results/round_6/datafiles/renum_fv_all_bindingenergies.csv")
# fulldelgb_df['Bound_Unbound'] = ['bound']*fulldelgb_df.shape[0]
# datastructure = final_af3_benchmark.merge(fulldelgb_df,on=['PDB_short','Seed','Model','Protein_type','Bound_Unbound'])
# # datastructure['Protein_type'] = ['nanobody' if datastructure.iloc[x].ocd==0.0 else 'antibody' for x in range(datastructure.shape[0])]
# datastructure= datastructure[(datastructure['Bound_Unbound']=='bound')&(datastructure['AF3_PDB'].str.contains('seed1'))&(datastructure['AF3_PDB'].str.contains('model_0'))]
# scorefxn = get_score_function()
datastructure = pd.read_csv("results/round_6/datafiles/topranked_benchmark_results.csv")
datastructure = datastructure[datastructure['Bound_Unbound']=='bound']
energy_fxn = "ref2015"
scorefxn = create_score_function(energy_fxn)
# i = 0
bes = []
pdbs = []
paths = []
ptypes = []
# for i in trange(1):
# while i<10:
for i in trange(datastructure.shape[0]):
    path_ = datastructure.iloc[i].AF3_Dir +  datastructure.iloc[i].AF3_PDB
    print(f"af3 vanilla pred path: {path_}")
    prottype = datastructure.iloc[i].Protein_type
    pose_1 = pose_from_pdb(path_)
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    if os.path.isdir(datastructure.iloc[i].AF3_Dir+'relaxed/'):
        overall_relax_pdb_dest = datastructure.iloc[i].AF3_Dir+'relaxed/overall_'+datastructure.iloc[i].AF3_PDB
        interface_relax_pdb_dest = datastructure.iloc[i].AF3_Dir+'relaxed/interface_'+datastructure.iloc[i].AF3_PDB
        print(overall_relax_pdb_dest)
    else:
        os.mkdir(datastructure.iloc[i].AF3_Dir+'relaxed/')
        overall_relax_pdb_dest = datastructure.iloc[i].AF3_Dir+'relaxed/overall_'+datastructure.iloc[i].AF3_PDB
        interface_relax_pdb_dest = datastructure.iloc[i].AF3_Dir+'relaxed/interface_'+datastructure.iloc[i].AF3_PDB
        print(overall_relax_pdb_dest)

    fr = FastRelax()
    fr.set_scorefxn(scorefxn)
    fr.max_iter(100)
    if not os.getenv("DEBUG"):
        fr.apply(pose_1)
    pose_1.dump_pdb(overall_relax_pdb_dest)
    # pose = pose_1.clone()
    # tf = TaskFactory()
    # tf.push_back(operation.InitializeFromCommandline())
    # tf.push_back(operation.RestrictToRepacking())
    # nbr_selector = selections.NeighborhoodResidueSelector()
    # try:
    #     cdr_selector = CDRResidueSelector()
    #     all_cdrs = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_all_cdrs(pose_i1)
    #     cdr_selector.set_cdrs(all_cdrs)
    #     nbr_selector.set_focus_selector(cdr_selector)
    #     # nbr_selector.set_include_focus_in_subset(True) #Default is true, so not including here.
    #     prevent_repacking_rlt = operation.PreventRepackingRLT()
    #     prevent_subset_repacking = operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    #     cdr_res = []
    #     print("CDR")
    #     for i in select.get_residue_set_from_subset(cdr_selector.apply(pose)):
    #         print(i)
    #         cdr_res.append(i)
            
    #     print("\nCDR+Neighbors")
    #     for i in select.get_residue_set_from_subset(nbr_selector.apply(pose)):
    #         if i in cdr_res:
    #             print(i,"CDR")
    #         else:
    #             print(i)
    #     pack_cdrs_and_neighbors_tf = tf.clone()
    #     mmf = MoveMapFactory()
    #     breakornot = True
    #     #mm_enable and mm_disable are Enums (numbered variables) that come when we import the MMF
    #     mmf.add_bb_action(mm_enable, cdr_selector) # do not want bb movement.

    #     # mmf.add_chi_action(mm_enable, cdr_selector) #We are taking this out for speed.
    #     mm  = mmf.create_movemap_from_pose(pose)
    #     fr.set_movemap_factory(mmf)
    #     fr.set_task_factory(pack_cdrs_and_neighbors_tf)
    #     if not os.getenv("DEBUG"):
    #         fr.apply(pose)
    #     pose.dump_pdb(interface_relax_pdb_dest)
        
    
    b_E_original = interface_energy_calc(path_,scorefxn,prottype)
    b_E_relaxed = interface_energy_calc(overall_relax_pdb_dest,scorefxn,prottype)
    bes.append(b_E_relaxed)
    pdbs.append('overall_'+datastructure.iloc[i].AF3_PDB)
    paths.append(datastructure.iloc[i].AF3_Dir+'relaxed/')
    ptypes.append(prottype)
    # b_E_interfacerelax = interface_energy_calc(interface_relax_pdb_dest,scorefxn,prottype)
    delta_e = b_E_original - b_E_relaxed
    # delta_e_interfaceonly = b_E_original - b_E_interfacerelax
    print(f"original energy: {b_E_original}")
    print(f"overall relaxed energy: {b_E_relaxed}")
    # print(f"CDRs + 6Å neighbors relaxed energy: {b_E_interfacerelax}")
    print(f"overall vs vanilla delta: {delta_e}")
    # print(f"interface vs vanilla delta: {delta_e_interfaceonly}")
    # except Exception as e:
    #     print(f"ERROR: {e}")
    #     breakornot = False
    #     print("faaaaaailed")
relaxed_bes_df = pd.DataFrame({"Pred_Dir":paths,"Pred_PDB":pdbs,"del_G_B":bes,"Protein_type":ptypes})
relaxed_bes_df.to_csv("results/round_6/datafiles/topranked_renum_fv_all_RELAXEDbindingenergies.csv")