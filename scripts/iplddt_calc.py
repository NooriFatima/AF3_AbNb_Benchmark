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


import pandas as pd
import numpy as np
import argparse
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.Superimposer import Superimposer
import urllib.request

# def get_interface_residues(pdb_file, partners, cutoff=10.0):
#     """Obtain the interface residues in a particular predicted structure. 
#     Args
#         pdb_file (str): Path to a PDB file
#         partners (str): String annotating the docking partners. Partners are 
#                         defined as <partner chain 1>_<partner chain 2>. for e.g.
#                         A_B, A_HL, ABC_DE, and so on. Note this is applicable for
#                         proteins and peptides with canonical amino acids only.
#         cutoff (float): Cut-off to determine the interface residues. Default is 
#                         10 Angstorms.
#     """
#     parser = PDBParser()
#     structure = parser.get_structure("model", pdb_file)

#     partner1 = partners.split("_")[0]
#     partner2 = partners.split("_")[1]

#     interface_residues = {'partner1': [], 'partner2': []}

#     for chain1 in partner1:
#         for chain2 in partner2:
#             chain1_residues = set(structure[0][chain1].get_residues())
#             chain2_residues = set(structure[0][chain2].get_residues())
#             int1_residues = []
#             int2_residues = []

#             for residue1 in chain1_residues:
#                 for residue2 in chain2_residues:
#                     distance = min(
#                         [np.linalg.norm(atom1.get_coord() - atom2.get_coord())
#                          for atom1 in residue1.get_atoms()
#                          for atom2 in residue2.get_atoms()])

#                     if distance <= cutoff:
#                         int1_residues.append((chain1,residue1))
#                         int2_residues.append((chain2,residue2))

#             interface_residues['partner1'].extend(list(set(int1_residues)))
#             interface_residues['partner2'].extend(list(set(int2_residues)))

#     interface_residues['partner1'] = list(set(interface_residues['partner1']))
#     interface_residues['partner2'] = list(set(interface_residues['partner2']))

#     return interface_residues

# def get_interface_residue_b_factors(pdb_file, partners, cutoff = 10):
#     """
#     Returns a dictionary of interface residues and their B-factor values.
#     :param pdb_file: The path to the PDB file.
#     :param chain1: The ID of the first chain.
#     :param chain2: The ID of the second chain.
#     :param cutoff_distance: The cutoff distance for identifying interface residues.
#     """
#     structure = PDBParser().get_structure('pdb', pdb_file)
#     model = structure[0]
#     b_factors = {}
    
#     interface_residues = get_interface_residues(pdb_file, partners, cutoff )
    
#     for k, v in interface_residues.items():
#         for residue in v:
#             for atom in residue[-1]:
#                 if ( atom.get_name() == 'CA' ):
#                     b_factors[(k, residue[0], residue[-1].get_id()[1], atom.get_name())] = atom.get_bfactor()
            
#     return b_factors

# def get_interface_res_fast(pdbfile,partners,cutoff=10.0):
#     """Obtain the interface residues in a particular predicted structure. 
#     Args
#         pdb_file (str): Path to a PDB file
#         partners (str): String annotating the docking partners. Partners are 
#                         defined as <partner chain 1>_<partner chain 2>. for e.g.
#                         A_B, A_HL, ABC_DE, and so on. Note this is applicable for
#                         proteins and peptides with canonical amino acids only.
#         cutoff (float): Cut-off to determine the interface residues. Default is 
#                         10 Angstorms.

#     * Importantly: assumes single ligand chain, multiple protein chains!!
#     """
#     ppdb1 = PandasPdb().read_pdb(pdbfile)
#     chain_order= ppdb1.df['ATOM'].chain_id.unique().tolist()
#     # total_chains = len(chain_order)
#     ligand_chain = partners.split("_")[-1]
#     ligand_idx = chain_order.index(ligand_chain)
#     prot_chains = [x for x in partners.split("_")[0]]
#     prot_idx = [chain_order.index(x) for x in prot_chains]
#     # print(prot_idx)
#     ca_df = ppdb1.df['ATOM'][ppdb1.df['ATOM']['atom_name']=='CA'].reset_index().drop(columns=['index'])
#     ca_coords = coord_extractor(ca_df)
#     edm = predicted_EDMGen(ca_coords).squeeze()
#     # print(edm.shape)
#     # print([[0,prot_chains[x]] for x in chain_order])
#     res_lens = [[0,ca_df[ca_df['chain_id']==x].shape[0]] for x in chain_order]
#     prot_res_raw = []
#     ligand_res_raw = []
#     ligand_res = []
#     prot_res = []
#     for idces in prot_idx:
#         # print(idces)
#         # print(prot_chains[idces])
#         ligand_indx_start = np.sum([res_lens[x][1] for x in range(ligand_idx)])
#         ligand_indx_end = np.sum([res_lens[x][1] for x in range(ligand_idx+1)])
#         if idces == 0:
#             prot_indx_start = 0
#         else:  
#             prot_indx_start = np.sum([res_lens[i][1] for i in range(idces)])
#         prot_indx_end = np.sum([res_lens[i][1] for i in range(idces+1)])
#         interface_edm = edm[prot_indx_start:prot_indx_end,ligand_indx_start:ligand_indx_end]
#         interface_res_prot,interface_res_ligand = torch.where(interface_edm<=cutoff)
#         toggled_dists = interface_edm[list(interface_res_prot.numpy()),list(interface_res_ligand.numpy())]
#         # print(toggled_dists)
#         true_prot_idx = pd.Series(interface_res_prot.numpy() + prot_indx_start).unique().tolist()
#         true_lig_idx = pd.Series(interface_res_ligand.numpy() + ligand_indx_start).unique().tolist()
#         # prot_plddts_raw = ca_df.loc[true_prot_idx,'b_factor'].tolist()
#         ligand_res.extend(set(true_lig_idx))
#         prot_res.extend(set(true_prot_idx))
#         # print("protein idx: {}".format(ca_df.iloc[list(true_prot_idx)].residue_number.tolist()))
#         # print("ligand idx: {}".format(ca_df.iloc[list(true_lig_idx)].residue_number.tolist()))
#         prot_res_raw.extend(ca_df.iloc[list(true_prot_idx)].residue_number.tolist())
#         ligand_res_raw.extend(ca_df.iloc[list(true_lig_idx)].residue_number.tolist())
    
#     prot_plddts = ca_df.loc[prot_res,'b_factor'].tolist()
#     prot_chainids = ca_df.loc[prot_res,'chain_id'].tolist()
#     prot_info = [(prot_chainids[x],'CA',prot_res_raw[x]) for x in range(len(prot_res_raw))]
#     lig_info = [(ligand_chain,'CA',ligand_res_raw[x]) for x in range(len(ligand_res_raw))]
#     prot_zip = zip(prot_info,prot_plddts)
#     plddts = dict(prot_zip)
#     lig_plddts = ca_df.loc[ligand_res,'b_factor'].tolist()
#     lig_zip = zip(lig_info,lig_plddts)
#     plddts.update(lig_zip)
#     i_plddt = np.mean(list(plddts.values()))
#     # print(i_plddt)
#     return plddts,i_plddt


datastructure = pd.read_csv("fixed_all_Ab_comparison_datastruc.csv").drop(columns=['Unnamed: 0'])
nb_datastruc = pd.read_csv("fixed_all_Nb_comparison_datastruc.csv")
datastructure = pd.concat([datastructure,nb_datastruc],keys=['antibody','nanobody']).reset_index().drop(columns=['level_1']).rename(columns={"level_0":"Protein_type"})



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
iplddt_df.to_csv("results/AF3_iplddts.csv")