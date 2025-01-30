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
    pyrosetta.init(init_string, silent=silent)
init_pyrosetta()

parser = argparse.ArgumentParser()
parser.add_argument("datafilepath", help="datastructure filepath")
parser.add_argument("resultsfilepath", help="filepath to dump results ")
args = parser.parse_args()


# run overall relax + regional (interface only) relax

datastructure = pd.read_csv(f"{args.datafilepath}")
datastructure = datastructure[datastructure['Bound_Unbound']=='bound']
energy_fxn = "ref2015"
scorefxn = create_score_function(energy_fxn)

bes = []
pdbs = []
paths = []
ptypes = []

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
        
    b_E_original = interface_energy_calc(path_,scorefxn,prottype)
    b_E_relaxed = interface_energy_calc(overall_relax_pdb_dest,scorefxn,prottype)
    bes.append(b_E_relaxed)
    pdbs.append('overall_'+datastructure.iloc[i].AF3_PDB)
    paths.append(datastructure.iloc[i].AF3_Dir+'relaxed/')
    ptypes.append(prottype)
    delta_e = b_E_original - b_E_relaxed
    print(f"original energy: {b_E_original}")
    print(f"overall relaxed energy: {b_E_relaxed}")
    print(f"overall vs vanilla delta: {delta_e}")
relaxed_bes_df = pd.DataFrame({"Pred_Dir":paths,"Pred_PDB":pdbs,"del_G_B":bes,"Protein_type":ptypes})
relaxed_bes_df.to_csv(f"{args.resultsfilepath}")