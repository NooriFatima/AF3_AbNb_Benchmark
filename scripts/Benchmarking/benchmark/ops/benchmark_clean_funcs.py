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
from Levenshtein import distance
from Bio.Align.Applications import ClustalwCommandline
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import matplotlib.pyplot as plt
from Bio import AlignIO
# from protein import *

#from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.all_funcs import *

from pyrosetta import *

def exists(x):
    return x is not None
def init_pyrosetta(init_string=None, silent=True):
    if not exists(init_string):
        init_string = "-mute all -ignore_zero_occupancy false -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string, silent=silent)
init_pyrosetta()

def pyrosetta_af3_benchmark(af3_dir_contents_df,native_dir_contents_df,PyRosetta_AF3_Nb_df,PyRosetta_AF3_Ab_df,Nb_df,Ab_df):
    err_log = []
    problem_dir = []
    problem_pdbs = []
    for i in trange(af3_dir_contents_df.shape[0]):
    # for i in trange(10):
        per_pdb_results = []
        pdb_names = []
        b_ub = []
        af3_pdb = af3_dir_contents_df.iloc[i].Files
        af3_base_dir = af3_dir_contents_df.iloc[i].Dir
        native_pdb_info = native_dir_contents_df[native_dir_contents_df['Files']==af3_pdb]
        print(af3_pdb)
        native_dir = native_pdb_info.Dir
        # print(native_dir)
        
        if native_dir.shape[0]>1:
            bool_= native_dir.str.contains(af3_base_dir.split('/')[8].lower()) #.iloc[1]
            native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
            if native_dir.shape[0]>1:
                if af3_base_dir.split('/')[9].lower() == 'nb': type_ = 'nanobody' ;
                bool_= native_dir.str.contains(type_) #.iloc[1]
                # print(bool_)
                native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
            else:
                pass
        else:
            pass
        native_dir = '/'.join(native_dir.item().split("/")[:7])+'/renamed/'+'/'.join(native_dir.item().split("/")[7:])
        if 'nanobody' in native_dir:
            df = Nb_df
            native_file = native_pdb_info.Files
            for j in range(1,4):
                seed_dir = af3_base_dir+'fold_'+af3_pdb+'_seed'+str(j)+'/'
                # for k in range(1):
                for k in range(0,5):
                    af3_basename= 'renamed_fold_'+af3_pdb+'_seed'+str(j)+'_model_'+str(k)
                    Run_or_not = run_check(af3_pdb,j,k,PyRosetta_AF3_Nb_df)
                    # print(Run_or_not)
                    if Run_or_not == False:
                        try:
                            native_file_ = native_dir+native_file.iloc[0]+".pdb"
                            pred_file = seed_dir+af3_basename+'.pdb'
                            native_dest = "/".join(native_file_.split('/')[:8])+'/reordered/'+"/".join(native_file_.split('/')[8:])
                            _ = reorder(native_file_,native_dest)
                            native_pose = pose_from_pdb(native_dest)
                            # native_pose = apply_align(native_pose)
                            # print(f'native sequence from pyrosetta: {native_pose.sequence()}')
                            # print(f'atm seq sequence: {get_atmseq(native_file_)}')
                            # print(f'seqres sequence: {get_seqres(native_file_)}')
                            pred_pose = pose_from_pdb(pred_file)
                            # pred_pose = apply_align(pred_pose)
                            # print(pred_pose.sequence())
                            # print(f'pred sequence from pyrosetta: {pred_pose.sequence()}')
                            # print(f'atm seq sequence: {get_atmseq(pred_file)}')
                            # print(f'seqres sequence: {get_seqres(pred_file)}')
                            pdb_result = get_ab_metrics(native_pose,pred_pose)
                            pdb_names.append(af3_basename)
                            b_ub.append(af3_base_dir.split('/')[8].lower())
                            print(pdb_result)
                            per_pdb_results.append(pdb_result)
                        except Exception as e:
                            err_log.append(e)
                            problem_pdbs.append(af3_basename)
                            problem_dir.append(seed_dir)
                            print(e)
                            break
                    else:
                        pass
            if len(per_pdb_results) == 0:
                pass
            else:
                pdb_df = pd.DataFrame(per_pdb_results)
                bub = pd.DataFrame(b_ub)
                pdb_info_df = pd.DataFrame([[ "_".join(x.split("_")[2:4]), "_".join(x.split("_")[3].split('d')[-1]), x[-1] ]for x in pdb_names],columns=['PDB','Seed','Model'])
                pdb_df = pd.concat([pdb_df,pdb_info_df,bub],axis=1)
                PyRosetta_AF3_Nb_df = pd.concat([PyRosetta_AF3_Nb_df,pdb_df],axis=0)
        else:
            df = Ab_df
            native_file = native_pdb_info.Files
            for j in range(1,4):
                seed_dir = af3_base_dir+'fold_'+af3_pdb+'_seed'+str(j)+'/'
                # for k in range(1):
                for k in range(0,5):
                    af3_basename= 'renamed_fold_'+af3_pdb+'_seed'+str(j)+'_model_'+str(k)
                    Run_or_not = run_check(af3_pdb,j,k,PyRosetta_AF3_Ab_df)
                    # print(Run_or_not)
                    if Run_or_not == False:
                        try:
                            native_file_ = native_dir+native_file.iloc[0]+".pdb"
                            pred_file = seed_dir+af3_basename+'.pdb'
                            native_dest = "/".join(native_file_.split('/')[:8])+'/reordered/'+"/".join(native_file_.split('/')[8:])
                            _ = reorder(native_file_,native_dest)
                            native_pose = pose_from_pdb(native_dest)
                            # native_pose = apply_align(native_pose)
                            # print(f'native sequence from pyrosetta: {native_pose.sequence()}')
                            # print(f'atm seq sequence: {get_atmseq(native_file_)}')
                            # print(f'seqres sequence: {get_seqres(native_file_)}')
                            pred_pose = pose_from_pdb(pred_file)
                            # pred_pose = apply_align(pred_pose)
                            # print(f'pred sequence from pyrosetta: {pred_pose.sequence()}')
                            # print(f'atm seq sequence: {get_atmseq(pred_file)}')
                            # print(f'seqres sequence: {get_seqres(pred_file)}')
                            pdb_result = get_ab_metrics(native_pose,pred_pose)
                            print(pdb_result)
                            per_pdb_results.append(pdb_result)
                            pdb_names.append(af3_basename)
                            b_ub.append(af3_base_dir.split('/')[8].lower())
                        except Exception as e:
                            err_log.append(e)
                            problem_pdbs.append(af3_basename)
                            problem_dir.append(seed_dir)
                            print(e)
                            break
                    else:
                        pass
            if len(per_pdb_results) == 0:
                pass
            else:
                pdb_df = pd.DataFrame(per_pdb_results)
                bub = pd.DataFrame(b_ub)
                pdb_info_df = pd.DataFrame([[ "_".join(x.split("_")[2:4]), "_".join(x.split("_")[3].split('d')[-1]), x[-1] ]for x in pdb_names],columns=['PDB','Seed','Model'])
                pdb_df = pd.concat([pdb_df,pdb_info_df,bub],axis=1)
                PyRosetta_AF3_Ab_df = pd.concat([PyRosetta_AF3_Ab_df,pdb_df],axis=0)
    return PyRosetta_AF3_Ab_df, PyRosetta_AF3_Nb_df

def pyrosetta_af2_benchmark(af2_pdbs,native_dir_contents_df,AF2_pred_dir,\
                            bound_Ab_seqdf,bound_Nb_seqdf,unbound_Ab_seqdf,unbound_Nb_seqdf,\
                                PyRosetta_Ab_results_AF2,PyRosetta_Nb_results_AF2,\
                                    af2_dest_dir_nb,af2_dest_dir_ab,\
                                        renamed_dest_dir_BAb,renamed_dest_dir_BNb,renamed_dest_dir_uAb,renamed_dest_dir_uNb):
    
    counter=0
    pdb_calcs = []
    pdbname = []
    b_ub = []
    prot_types = []
    for i in trange(len(af2_pdbs)):
        # print(af2_pdbs[i])
        short_name = "_".join(af2_pdbs[i].split("_")[:2])
        if short_name.split("_")[0] != '7tn9':
            print(short_name)
            native_pdb_info = native_dir_contents_df[native_dir_contents_df['Files']==short_name]
            native_dir = native_pdb_info.Dir #.item()
            # print(native_dir)
            # per_pdb_results = []
            # pdb_names = []
            # b_ub = []
            b_ub_af2 = af2_pdbs[i].split("_")[2]
            # print(b_ub_af2)
            
            # print(AF2_pred_dir+i)
            seqres = get_atmseq(AF2_pred_dir+af2_pdbs[i])
            # print(len(seqres))
            if b_ub_af2 == 'bound' and len(seqres)==3:
                df=bound_Ab_seqdf
                pred_dir = af2_dest_dir_ab
                prot_type = 'Ab'
                Run_or_not = run_check_af2(short_name,PyRosetta_Ab_results_AF2)
            elif b_ub_af2 == 'bound' and len(seqres)==2:
                df=bound_Nb_seqdf
                prot_type = 'Nb'
                pred_dir = af2_dest_dir_nb
                Run_or_not = run_check_af2(short_name,PyRosetta_Nb_results_AF2)
            elif b_ub_af2 == 'unbound' and len(seqres)==2:
                df=unbound_Ab_seqdf
                pred_dir = af2_dest_dir_ab
                prot_type = 'Ab'
                Run_or_not = run_check_af2(short_name,PyRosetta_Ab_results_AF2)
            else:
                df=unbound_Nb_seqdf
                pred_dir = af2_dest_dir_nb
                prot_type = 'Nb'
                Run_or_not = run_check_af2(short_name,PyRosetta_Nb_results_AF2)
            if native_dir.shape[0]>1:
                bool_= native_dir.str.contains(b_ub_af2) #.iloc[1]

                native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
                if native_dir.shape[0]>1:
                    # if af3_base_dir.split('/')[9].lower() == 'nb': type_ = 'nanobody' ;
                    if b_ub_af2 == 'bound' and len(seqres)==3:
                        native_dir=renamed_dest_dir_BAb
        
                    elif b_ub_af2 == 'bound' and len(seqres)==2:
                        native_dir=renamed_dest_dir_BNb
                        
                    elif b_ub_af2 == 'unbound' and len(seqres)==2:
                        native_dir=renamed_dest_dir_uAb
                        
                    else:
                        native_dir = renamed_dest_dir_uNb
                        
            else:
                # print(native_dir)
                native_dir = native_dir.item()
                native_dir = "/".join(native_dir.split("/")[:7])+'/renamed/'+"/".join(native_dir.split("/")[7:])
            # print(native_dir)
            if Run_or_not == False:
                # pdb_names.append(short_name)
                # b_ub.append(b_ub_af2)
                # if (b_ub_af2 == 'bound' and len(seqres)==3) or (b_ub_af2 == 'unbound' and len(seqres)==2): 
                #     pass
                # elif( b_ub_af2 == 'bound' and len(seqres)==2) or (b_ub_af2 == 'unbound' and len(seqres)==1):
                #     pass
                try:
                    native_pose = pose_from_pdb(native_dir+short_name+'.pdb')
                    pred_pose = pose_from_pdb(pred_dir+af2_pdbs[i])
                    rmsd_calcs = get_ab_metrics(native_pose,pred_pose)
                    print(rmsd_calcs)
                    prot_types.append(prot_type)
                    pdbname.append(short_name)
                    b_ub.append(b_ub_af2)
                    per_pdb = pd.DataFrame(rmsd_calcs,index=[counter])
                    pdb_calcs.append(per_pdb)
                    counter+=1
                    if (counter > 0) and (len(pdb_calcs)==0):
                        print("something is off!")
                        break
                except Exception as e:
                    print(e)

    return PyRosetta_Ab_results_AF2,PyRosetta_Nb_results_AF2

def simple_benchmark(pdb1,pdb2):
    pdb1_pose = pose_from_pdb(pdb1)
    pdb2_pose = pose_from_pdb(pdb2)
    rmsd_calcs = get_ab_metrics(pdb1_pose,pdb2_pose)
    per_pdb = pd.DataFrame(rmsd_calcs,index=[0])
    return per_pdb

def query_af3_info(af3_dir_contents_df,i):
    seed_dirs = []
    basenames = []
    af3_pdb = af3_dir_contents_df.iloc[i].Files
    af3_base_dir = af3_dir_contents_df.iloc[i].Dir
    for j in range(1,4):
        seed_dir = af3_base_dir+'fold_'+af3_pdb+'_seed'+str(j)+'/'
        seed_dirs.append(seed_dir)
        for k in range(0,5):
            af3_basename= 'renamed_fold_'+af3_pdb+'_seed'+str(j)+'_model_'+str(k)
            basenames.append(af3_basename)
    return seed_dirs, basenames, af3_pdb

def query_native_info(native_dir_contents_df,pdb,protein_type,b_ub):
    native_pdb_info = native_dir_contents_df[native_dir_contents_df['Files']==pdb]
    native_file = native_pdb_info.Files.iloc[0]
    native_dir = native_pdb_info.Dir
    if native_dir.shape[0]>1:
        bool_= native_dir.str.contains(b_ub) #.iloc[1]
        native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
        if native_dir.shape[0]>1:
            bool_= native_dir.str.contains(protein_type) #.iloc[1]
            native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
        else:
            pass
    else:
        pass
    native_dir = '/'.join(native_dir.item().split("/")[:7])+'/renamed/'+'/'.join(native_dir.item().split("/")[7:])
    native_file_ = native_dir+native_file+".pdb"
    native_dest = "/".join(native_file_.split('/')[:8])+'/reordered/'+"/".join(native_file_.split('/')[8:])
    if os.path.isfile(native_dest):
        pass
    else:
        _ = reorder(native_file_,native_dest)
    
    return native_dest

# def query_igfold_info(igfold_dir_contents_df,i):

#     return

# def query_af2_info(af2_dir_contents_df,i):
    
#     return

# def get_pdb_info(i,pdb_source,native_dir_contents_df,interest_dir_contents_df):
#     pdb = interest_dir_contents_df.iloc[i].Files
#     pdb_base_dir = interest_dir_contents_df.iloc[i].Dir
#     native_pdb_info = native_dir_contents_df[native_dir_contents_df['Files']==pdb]
#     native_dir = native_pdb_info.Dir
#     if native_dir.shape[0]>1:
#         bool_= native_dir.str.contains(pdb_base_dir.split('/')[8].lower()) #.iloc[1]
#         native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
#         if native_dir.shape[0]>1:
#             if pdb_base_dir.split('/')[9].lower() == 'nb': type_ = 'nanobody' ;
#             bool_= native_dir.str.contains(type_) #.iloc[1]
#             # print(bool_)
#             native_dir=native_dir_contents_df.loc[bool_.index[bool_==True],'Dir']
#         else:
#             pass
#     else:
#         pass
#     native_dir = '/'.join(native_dir.item().split("/")[:7])+'/renamed/'+'/'.join(native_dir.item().split("/")[7:])
#     return
