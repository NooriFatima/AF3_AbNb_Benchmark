from biopandas.pdb import PandasPdb
import pandas as pd
import requests
import time
import os
import numpy as np
# import Bio
from abnumber import Chain
# import math
import enum
import torch
import torch.nn.functional as F
from tqdm import trange
from Bio.PDB import PDBParser
# from Bio.PDB.Selection import unfold_entities
from Bio.SeqIO import PdbIO
# import warnings
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Sequence, List, Optional, Union
# from Levenshtein import distance
# from Bio.Align.Applications import ClustalwCommandline
from Bio import SeqIO
# from Bio.Seq import Seq
# from Bio.SeqRecord import SeqRecord
# import matplotlib.pyplot as plt
# from Bio import AlignIO
# from protein import *
# from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.benchmark_clean_funcs import *
from pyrosetta import *
def exists(x):
    return x is not None
def init_pyrosetta(init_string=None, silent=True):
    if not exists(init_string):
        init_string = "-mute all -ignore_zero_occupancy false -detect_disulf true -detect_disulf_tolerance 1.5 -check_cdr_chainbreaks false"
    pyrosetta.init(init_string, silent=silent)
init_pyrosetta()


class BBHeavyAtom(enum.IntEnum):
    N = 0; CA = 1; C = 2; O = 3; CB = 4; OXT=14;
def dihedral_from_four_points(p0, p1, p2, p3):
    """
    Args:
        p0-3:   (*, 3).
    Returns:
        Dihedral angles in radian, (*, ).
    """
    v0 = p2 - p1
    v1 = p0 - p1
    v2 = p3 - p2
    u1 = torch.cross(v0, v1, dim=-1)
    n1 = u1 / torch.linalg.norm(u1, dim=-1, keepdim=True)
    u2 = torch.cross(v0, v2, dim=-1)
    n2 = u2 / torch.linalg.norm(u2, dim=-1, keepdim=True)
    sgn = torch.sign( (torch.cross(v1, v2, dim=-1) * v0).sum(-1) )
    dihed = sgn*torch.acos( (n1 * n2).sum(-1).clamp(min=-0.999999, max=0.999999) )
    dihed = torch.nan_to_num(dihed)
    return dihed

def get_backbone_dihedral_angles(pos_atoms):
    """
    Args:
        pos_atoms:  (N, L, A, 3).
    Returns:
        bb_dihedral:    Omega, Phi, and Psi angles in radian, (N, L, 3).
    """
    pos_N  = pos_atoms[:, :, BBHeavyAtom.N]   # (N, L, 3)
    pos_CA = pos_atoms[:, :, BBHeavyAtom.CA]
    pos_C  = pos_atoms[:, :, BBHeavyAtom.C]

    # N-termini don't have omega and phi
    omega = F.pad(
        dihedral_from_four_points(pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:]), 
        pad=(1, 0), value=0,
    )
    phi = F.pad(
        dihedral_from_four_points(pos_C[:, :-1], pos_N[:, 1:], pos_CA[:, 1:], pos_C[:, 1:]),
        pad=(1, 0), value=0,
    )

    # C-termini don't have psi
    psi = F.pad(
        dihedral_from_four_points(pos_N[:, :-1], pos_CA[:, :-1], pos_C[:, :-1], pos_N[:, 1:]),
        pad=(0, 1), value=0,
    )

    bb_dihedral = torch.stack([omega, phi, psi], dim=-1)
    return bb_dihedral

#OLD
def kabsch(A, B):
        a_mean = A.mean(dim=1, keepdims=True).type('torch.DoubleTensor')
        b_mean = B.mean(dim=1, keepdims=True).type('torch.DoubleTensor')
        A_c = A - a_mean
        B_c = B - b_mean
        # Covariance matrix
        H = torch.bmm(A_c.transpose(1,2), B_c)  # [B, 3, 3]
        U, S, V = torch.svd(H)
        # Rotation matrix
        R = torch.bmm(V, U.transpose(1,2))  # [B, 3, 3]
        # Translation vector
        t = b_mean - torch.bmm(R, a_mean.transpose(1,2)).transpose(1,2)
        A_aligned = torch.bmm(R, A.transpose(1,2)).transpose(1,2) + t
        return A_aligned, R, t

def get_handle(fp: Path, extension: str):
    """
    Args:
        fp: (Path) file path
        extension: (str) ".gz" or other extensions

    Returns:
        handler: file handler
    """
    handle = None
    if extension == ".gz":
        import gzip
        handle = gzip.open(fp, "rt")
    else:
        handle = open(fp, "r")

    return handle

def get_single_copy_HLA(ppdb,H_id,L_id,A_id):

    sep_chain = PandasPdb()
    sep_chain = ppdb
    sep_chain.df['ATOM']=ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id']==H_id) | (ppdb.df['ATOM']['chain_id']==L_id) | (ppdb.df['ATOM']['chain_id']==A_id)]
    
    return sep_chain

def get_single_copy_HL(ppdb,H_id,L_id):

    sep_chain = PandasPdb()
    sep_chain = ppdb
    sep_chain.df['ATOM']=ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id']==H_id) | (ppdb.df['ATOM']['chain_id']==L_id)]
    
    return sep_chain

def get_single_copy_H(ppdb,H_id):

    sep_chain = PandasPdb()
    sep_chain = ppdb
    sep_chain.df['ATOM']=ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id']==H_id)]
    
    return sep_chain

def get_seq(ppdb, chain_ids):
    sequence = ppdb.amino3to1()
    seqs = []
    for i in chain_ids:
        seq = ''.join(sequence.loc[sequence['chain_id'] == i, 'residue_name'])
        seqs.append(seq)
    return seqs

def get_bb_coords(sep_chain):
    bb_only = PandasPdb()
    bb_only.df['ATOM'] = sep_chain.df['ATOM'][(sep_chain.df['ATOM']['atom_name']=='N') | (sep_chain.df['ATOM']['atom_name']=='CA') | (sep_chain.df['ATOM']['atom_name']=='C')]
    bb_coords = torch.cat([torch.tensor(bb_only.df['ATOM']['x_coord'].tolist()).unsqueeze(0).T,\
                    torch.tensor(bb_only.df['ATOM']['y_coord'].tolist()).unsqueeze(0).T,\
                        torch.tensor(bb_only.df['ATOM']['z_coord'].tolist()).unsqueeze(0).T],dim=1)
    return bb_coords

def get_cdrs(seq):
    try:
        seq_ = Chain(seq,scheme='chothia')
        good_chain = True
        cdr_list = [seq_.cdr1_seq,seq_.cdr2_seq,seq_.cdr3_seq]
    except Exception as e:
        print(e)
        good_chain = False
    return good_chain, cdr_list

def get_seqres(struct_fp,extension='.pdb'):

    handle = get_handle(fp=struct_fp, extension=extension)
    seqres = {i.id[-1]: str(i.seq) for i in PdbIO.PdbSeqresIterator(handle)}
    handle.close()

    return seqres

def get_atmseq(struct_fp,extension='.pdb'):

    handle = get_handle(fp=struct_fp, extension=extension)
    atmseq = {i.id[-1]: str(i.seq).replace("X", "") for i in
                    PdbIO.PdbAtomIterator(handle)}  
    handle.close()

    return atmseq

def check_missing_res(atmseq,seqres,chain_id,indices):
    
    missing_res = [True if atmseq[chain_id][y[0]:y[1]] != seqres[chain_id][y[0]:y[1]] else False for y in indices]
    if True not in missing_res:
        return False
    else:
        return True

def run_check_af3(af3_pdb,j,k,df):
    if df.shape[0] == 0:
        return False
    else:
        df_query = df[df['PDB']==af3_pdb]
        # print("df query: {}".format(df_query))
        if df_query.shape[0] == 0:
            return False
        else:
            # print(f"seed: {j}")
            # print(f"model: {k}")
            file_query = df_query[(df_query['Seed'] == str(j)) & (df_query['Model'] == str(k))]
            # print("file query: {}".format(file_query))
            if file_query.shape[0] == 0:
                return False
            else:
                return True
            
def run_check_af2(af2_pdb,df):
    if df.shape[0] == 0:
        return False
    else:
        df_query = df[df['PDB']==af2_pdb]
        # print("df query: {}".format(df_query))
        if df_query.shape[0] == 0:
            return False
        else:  
            return True
    
def extract_mmcif_CA_data(dir_,pdbfile,df):
    ppdb = PandasMmcif()
    print(dir_+pdbfile)
    if os.path.isfile(dir_+pdbfile):
        ppdb.read_mmcif(dir_+pdbfile)
        # print(dir_+pdbfile)
        # print(ppdb.df['ATOM'])
        chain_ids = ppdb.df['ATOM']['auth_asym_id'].unique().tolist()
        sequence = ppdb.amino3to1()
        # print(df[(df["PDB"]==pdbfile)].HeavyChain)
        H_seq = df[(df["PDB"]==pdbfile)].HeavyChain.item()
        Hchain = Chain(H_seq, scheme='chothia')
        H_loops = [Hchain.cdr1_seq, Hchain.cdr2_seq, Hchain.cdr3_seq]
        L_seq = df[(df["PDB"]==pdbfile)].LightChain.item()
        Lchain = Chain(L_seq, scheme='chothia')
        L_loops = [Lchain.cdr1_seq, Lchain.cdr2_seq, Lchain.cdr3_seq]

        chain_order = []
        real_chain_ids =[]
        for i in chain_ids:
            chain_seq = ''.join(sequence.loc[sequence['auth_asym_id'] == i,'auth_comp_id'])
            if chain_seq == H_seq:
                # print("H matching: ")
                # print("from df: {}".format(H_seq))
                # print("from pdb: {}".format(chain_seq))
                chain_order.append('H')
                real_chain_ids.append(i)
                Hseq = chain_seq
            elif chain_seq == L_seq:
                # print("L matching: ")
                # print("from df: {}".format(L_seq))
                # print("from pdb: {}".format(chain_seq))
                chain_order.append('L')
                real_chain_ids.append(i)
                Lseq = chain_seq
            else:
                pass
        chain_dict= dict(zip(chain_order,real_chain_ids))
        print(chain_dict)
        ppdb.df['ATOM'] = ppdb.df['ATOM'].loc[(ppdb.df['ATOM']['auth_atom_id']=='CA')]
    else:
        print(f"{dir_+pdbfile} does not exist")

    return ppdb, chain_dict, H_loops, L_loops, Hseq, Lseq

def extract_pdb_CA_data_AF2(dir_,pdbfile,df):
    ppdb = PandasPdb()
    # print(dir_+pdbfile)
    if "seed" in pdbfile:
        short_name= "_".join(pdbfile.split("_")[:2])+".pdb"
    else:
       short_name = pdbfile
    try:
        dir_ = dir_.item()
    except:
        pass
    if os.path.isfile(dir_+pdbfile):
        ppdb.read_pdb(dir_+pdbfile)
        # print(ppdb.df['ATOM']['chain_id'].unique().tolist())
        chain_ids = ppdb.df['ATOM']['chain_id'].unique().tolist()
        sequence = ppdb.amino3to1()
        # print(df[(df["PDB"]==short_name)].HeavyChain)
        H_seq = df[(df["PDB"]==short_name)].HeavyChain.item()
        # print(H_seq)
        Hchain = Chain(H_seq, scheme='chothia')
        H_loops = [Hchain.cdr1_seq, Hchain.cdr2_seq, Hchain.cdr3_seq]
        if 'LightChain' not in df.columns:
            L_seq = None
        else:
            L_seq = df[(df["PDB"]==short_name)].LightChain.item()
            Lchain = Chain(L_seq, scheme='chothia')
            L_loops = [Lchain.cdr1_seq, Lchain.cdr2_seq, Lchain.cdr3_seq]

        chain_order = []
        real_chain_ids =[]
        for i in chain_ids:
            # print(i)
            # print(L_seq)
            chain_seq = ''.join(sequence.loc[sequence['chain_id'] == i,'residue_name'])
            # print(chain_seq)
            if chain_seq[1:-1] in H_seq:
                # print("H matching: ")
                # print("from df: {}".format(H_seq))
                # print("from pdb: {}".format(chain_seq))
                chain_order.append('H')
                real_chain_ids.append(i)
                Hseq = chain_seq
            elif (L_seq != None):
                if(chain_seq[1:-1] in L_seq):
                    # print("L matching: ")
                    # print("from df: {}".format(L_seq))
                    # print("from pdb: {}".format(chain_seq))
                    chain_order.append('L')
                    real_chain_ids.append(i)
                    Lseq = chain_seq
                else: 
                    pass
            else:
                pass
        chain_dict= dict(zip(chain_order,real_chain_ids))
        # print(chain_dict)
        ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df['ATOM']['atom_name']=='CA')&((ppdb.df['ATOM']['alt_loc']=='') | (ppdb.df['ATOM']['alt_loc']=='A'))]
        # print(ppdb.df['ATOM'].iloc[:5])
        if 'LightChain' not in df.columns:
            return ppdb, chain_dict, H_loops,Hseq
        else:
            return ppdb, chain_dict, H_loops, L_loops, Hseq, Lseq
    else:
        print(f"{dir_+pdbfile} does not exist")

def extract_pdb_CA_data_AF3(dir_,pdbfile,df):
    ppdb = PandasPdb()
    # print(dir_+pdbfile)
    if "seed" in pdbfile:
        short_name= "_".join(pdbfile.split("_")[1:3])+".pdb"
    else:
       short_name = pdbfile
    if os.path.isfile(dir_+pdbfile):
        ppdb.read_pdb(dir_+pdbfile)
        # print(ppdb.df['ATOM']['chain_id'].unique().tolist())
        chain_ids = ppdb.df['ATOM']['chain_id'].unique().tolist()
        sequence = ppdb.amino3to1()
        # print(df[(df["PDB"]==short_name)].HeavyChain)
        H_seq = df[(df["PDB"]==short_name)].HeavyChain.item()
        # print(H_seq)
        Hchain = Chain(H_seq, scheme='chothia')
        H_loops = [Hchain.cdr1_seq, Hchain.cdr2_seq, Hchain.cdr3_seq]
        if 'LightChain' not in df.columns:
            L_seq = None
        else:
            L_seq = df[(df["PDB"]==short_name)].LightChain.item()
            Lchain = Chain(L_seq, scheme='chothia')
            L_loops = [Lchain.cdr1_seq, Lchain.cdr2_seq, Lchain.cdr3_seq]

        chain_order = []
        real_chain_ids =[]
        for i in chain_ids:
            # print(i)
            # print(L_seq)
            chain_seq = ''.join(sequence.loc[sequence['chain_id'] == i,'residue_name'])
            # print(chain_seq)
            if chain_seq[1:-1] in H_seq:
                # print("H matching: ")
                # print("from df: {}".format(H_seq))
                # print("from pdb: {}".format(chain_seq))
                chain_order.append('H')
                real_chain_ids.append(i)
                Hseq = chain_seq
            elif (L_seq != None):
                if(chain_seq[1:-1] in L_seq):
                    # print("L matching: ")
                    # print("from df: {}".format(L_seq))
                    # print("from pdb: {}".format(chain_seq))
                    chain_order.append('L')
                    real_chain_ids.append(i)
                    Lseq = chain_seq
                else: 
                    pass
            else:
                pass
        chain_dict= dict(zip(chain_order,real_chain_ids))
        # print(chain_dict)
        ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df['ATOM']['atom_name']=='CA')&((ppdb.df['ATOM']['alt_loc']=='') | (ppdb.df['ATOM']['alt_loc']=='A'))]
        # print(ppdb.df['ATOM'].iloc[:5])
        if 'LightChain' not in df.columns:
            return ppdb, chain_dict, H_loops,Hseq
        else:
            return ppdb, chain_dict, H_loops, L_loops, Hseq, Lseq
    else:
        print(f"{dir_+pdbfile} does not exist")

def coord_extractor(df):
    X = df['x_coord'].tolist()
    Y = df['y_coord'].tolist()
    Z = df['z_coord'].tolist()
    coords = torch.tensor([X,Y,Z]).transpose(0,1).unsqueeze(0)
    return coords

# def rmsd_(pred,true):
#     pred_aligned,_,_ = kabsch(pred.type(torch.float64),true.type(torch.float64))
#     #print("true coords: {}".format(true))
#     #print("aligned coords: {}".format(pred_aligned))
#     rmsd = torch.mean(torch.sqrt(torch.sum((pred_aligned-true).pow(2),-1)),-1)
#     return rmsd

def rmsd_(pred,true):
    pred_aligned,_,_ = kabsch(pred.type(torch.float64),true.type(torch.float64))
    # print("first 5 true coords: {}".format(true[:,:5,:]))
    # print("first 5 unaligned coords: {}".format(pred[:,:5,:]))
    # print("first 5 aligned coords: {}".format(pred_aligned[:,:5,:]))

    # print("last 5 true coords: {}".format(true[:,-5:,:]))
    # print("last 5 unaligned coords: {}".format(pred[:,-5:,:]))
    # print("last 5 aligned coords: {}".format(pred_aligned[:,-5:,:]))

    rmsd = torch.mean(torch.sqrt(torch.sum((pred_aligned-true).pow(2),-1)),-1)
    pRMSD = torch.sqrt(torch.sum((pred_aligned-true).pow(2),-1))
    return rmsd, pRMSD

def local_rmsd_(pred,true,pred_idx,true_idx):
    pred = pred[:,pred_idx[0]:pred_idx[1],:]
    true = true[:,true_idx[0]:true_idx[1],:]
    rmsd,_ = rmsd_(pred,true)
    return rmsd

def global_rmsd(pred,true,pred_idces):
    chain_rmsd,pRMSD = rmsd_(pred,true)
    # print("shape of chain rmsd: {}".format(chain_rmsd.shape))
    loop_rmsd = [[torch.mean(pRMSD[:,x[0]:x[1]],-1)] for x in pred_idces]
    # print(loop_rmsd)
    return chain_rmsd, loop_rmsd


def all_rmsds(pred_dir,true_dir,pred,true,df):
    pdb_pair = [(true_dir,true),(pred_dir,pred)]
    if 'LightChain' not in df.columns:
        true_df, true_chaindict, trueHloops, Hseq_true = extract_pdb_CA_data_AF2(true_dir,true,df)
        # print("Native H chain seq: {}".format(Hseq_true))
        # print("Length of native H chain seq: {}".format(len(Hseq_true)))
        pred_df, pred_chaindict, predHloops, Hseq_pred = extract_pdb_CA_data_AF2(pred_dir,pred,df)
        # print("Pred H chain seq: {}".format(Hseq_pred))
        # print("Length of predicted H chain seq: {}".format(len(Hseq_pred)))
        # H_min_length = np.min([true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']].shape[0],pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']].shape[0]])
        H_loopRMSDs = []
        Hloop_trueindex = [[Hseq_true.index(x), (Hseq_true.index(x)+len(x))] for x in trueHloops]
        Hloop_predindex = [[Hseq_pred.index(x), (Hseq_pred.index(x)+len(x))] for x in predHloops]
        H_truecoords = coord_extractor(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']])
        H_predcoords = coord_extractor(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']])

        # H_truecoords = coord_extractor(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']].iloc[:H_min_length,:])
        # H_predcoords = coord_extractor(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']].iloc[:H_min_length,:])

        H_rmsd, H_globalLoop_rmsd = global_rmsd(H_predcoords,H_truecoords,Hloop_predindex)
        for i in range(3):
            Hloop_rmsd = local_rmsd_(H_predcoords,H_truecoords,Hloop_predindex[i],Hloop_trueindex[i])
            # print(f"local H rmsds: {Hloop_rmsd}")
            H_loopRMSDs.append(Hloop_rmsd)
        
        return pdb_pair, H_rmsd, H_loopRMSDs,H_globalLoop_rmsd, None,None,None,None
    else:
        true_df, true_chaindict, trueHloops, trueLloops, Hseq_true, Lseq_true = extract_pdb_CA_data_AF2(true_dir,true,df)
        pred_df, pred_chaindict, predHloops, predLloops, Hseq_pred, Lseq_pred = extract_pdb_CA_data_AF2(pred_dir,pred,df)
        # H_true = PandasPdb()
        # H_true.df['ATOM']=true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']].iloc[:110,:]
        # H_true.to_pdb(path=f'./H_only/{true}', records=None, gz=False, append_newline=True)

        # H_pred = PandasPdb()
        # H_pred.df['ATOM']=pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']].iloc[:110,:]
        # H_pred.to_pdb(path=f'./H_only/{pred}', records=None, gz=False, append_newline=True)

        # H_min_length = np.min([true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']].shape[0],pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']].shape[0]])
        # L_min_length = np.min([true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['L']].shape[0],pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['L']].shape[0]])
        #print("H_true df : {}".format(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']]))
        # H_truecoords = coord_extractor(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']].iloc[:H_min_length,:])
        # L_truecoords = coord_extractor(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['L']].iloc[:L_min_length,:])

        # H_predcoords = coord_extractor(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']].iloc[:H_min_length,:])
        # #print("H_pred df : {}".format(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']]))
        # L_predcoords = coord_extractor(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['L']].iloc[:L_min_length,:])
        
        H_truecoords = coord_extractor(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['H']])
        L_truecoords = coord_extractor(true_df.df['ATOM'][true_df.df['ATOM']['chain_id'] == true_chaindict['L']])

        H_predcoords = coord_extractor(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']])
        #print("H_pred df : {}".format(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['H']]))
        L_predcoords = coord_extractor(pred_df.df['ATOM'][pred_df.df['ATOM']['chain_id'] == pred_chaindict['L']])

        # H_truecoords = H_truecoords[:,:110,:]
        # H_predcoords = H_predcoords[:,:110,:]
        # L_truecoords = L_truecoords[:,:110,:]
        # L_predcoords = L_predcoords[:,:110,:]

        #Superimpose each chain and then do RMSD

        H_loopRMSDs = []
        L_loopRMSDs = []
        Hloop_trueindex = [[Hseq_true.index(x), (Hseq_true.index(x)+len(x))] for x in trueHloops]
        Hloop_predindex = [[Hseq_pred.index(x), (Hseq_pred.index(x)+len(x))] for x in predHloops]

        Lloop_trueindex = [[Lseq_true.index(x), (Lseq_true.index(x)+len(x))] for x in trueLloops]
        Lloop_predindex = [[Lseq_pred.index(x), (Lseq_pred.index(x)+len(x))] for x in predLloops]

        #Get chain and global loop RMSDs

        # H_rmsd = rmsd_(H_predcoords.type(torch.float64),H_truecoords.type(torch.float64))
        # L_rmsd = rmsd_(L_predcoords.type(torch.float64),L_truecoords.type(torch.float64))
    
        H_rmsd, H_globalLoop_rmsd = global_rmsd(H_predcoords,H_truecoords,Hloop_predindex)
        L_rmsd, L_globalLoop_rmsd = global_rmsd(L_predcoords,L_truecoords,Lloop_predindex)

        #Get local loop RMSDs

        for i in range(3):
            Hloop_rmsd = local_rmsd_(H_predcoords,H_truecoords,Hloop_predindex[i],Hloop_trueindex[i])
            H_loopRMSDs.append(Hloop_rmsd)
            Lloop_rmsd = local_rmsd_(L_predcoords,L_truecoords,Lloop_predindex[i],Lloop_trueindex[i])
            L_loopRMSDs.append(Lloop_rmsd)
        if H_predcoords.shape[1]<150:
            return pdb_pair, H_rmsd, L_rmsd, H_loopRMSDs, L_loopRMSDs,H_globalLoop_rmsd,L_globalLoop_rmsd, None, None, None, None
        else:
            H_rmsd_noC, H_globalLoop_rmsd_noC = global_rmsd(H_predcoords[:,:128,:],H_truecoords[:,:128,:],Hloop_predindex)
            L_rmsd_noC, L_globalLoop_rmsd_noC = global_rmsd(L_predcoords[:,:128,:],L_truecoords[:,:128,:],Lloop_predindex)

            return pdb_pair, H_rmsd, L_rmsd, H_loopRMSDs, L_loopRMSDs,H_globalLoop_rmsd,L_globalLoop_rmsd, H_rmsd_noC, L_rmsd_noC, H_globalLoop_rmsd_noC, L_globalLoop_rmsd_noC


def convert_mmcif_pdb(src,dest):
    os.system(f"gemmi convert {src} {dest}")

def melt_Ab_results(benchmark_results_df):
    melted_df = pd.melt(benchmark_results_df, id_vars=['PDB','Seed','Model'], \
                        value_vars=['H_Fv','H_Fv_C','H1_local','H2_local','H3_local','H1_Fv','H2_Fv','H3_Fv','H1_Fv_C','H2_Fv_C','H3_Fv_C','L_Fv','L_Fv_C','L1_local','L2_local','L3_local','L1_Fv','L2_Fv','L3_Fv','L1_Fv_C','L2_Fv_C','L3_Fv_C'],\
                            var_name='Fv_Region', value_name='RMSD')
    return melted_df 

def melt_Nb_results(benchmark_results_df):
    melted_df = pd.melt(benchmark_results_df, id_vars=['PDB','Seed','Model'], \
                        value_vars=['H','H1_local','H2_local','H3_local','H1_global','H2_global','H3_global'],\
                            var_name='Fv_Region', value_name='RMSD')
    return melted_df 

def fasta_to_df(fasta_file):
    with open(fasta_file) as fasta:
        identifiers = []
        aligned_seqs = []
        for seq_record in SeqIO.parse(fasta, 'fasta'):  # (generator)
            identifiers.append(seq_record.id)
            aligned_seqs.append("".join(seq_record.seq))
    fasta.close()
    df = pd.DataFrame({"PDB":[x.split("|")[0] for x in identifiers],"Dir":[x.split("|")[1] for x in identifiers],"sequence":aligned_seqs})
    return df

def plot_region_rmsd(melted_df,protein_type):
    sns.set_theme(rc={'figure.figsize':(11.7,8.27)})
    # sns.set(font="Arial")
    nb_plot = sns.boxplot(data=melted_df, x='Fv_Region', y='RMSD',palette='Spectral')
    nb_plot.set_title(f"{protein_type} Structure Prediction RMSD")
    nb_plot.axhline(y=1)
    # plt.savefig("plots/Nb_structure_prediction.png",dpi=300,transparent=True)
    return

def reorder_rename_Ab(summary_df,pdb,pdb_src,pdb_dest):
    pdb_fvs = summary_df #.loc[summary_df["pdb"]==pdb]
    H_chains = pdb_fvs.Hchain #.tolist()
    L_chains = pdb_fvs.Lchain #.tolist()
    for j in range(len(H_chains)):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_src)
        ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df["ATOM"]['chain_id'] == H_chains[j]) | (ppdb.df["ATOM"]['chain_id'] == L_chains[j])]
        ppdb.df['ATOM'].loc[ppdb.df["ATOM"]["chain_id"] == H_chains[j],"chain_id"] = "H"
        ppdb.df['ATOM'].loc[ppdb.df["ATOM"]["chain_id"] == L_chains[j],"chain_id"] = "L"
        indices_H = ppdb.df['ATOM'][ppdb.df["ATOM"]['chain_id'] == "H"].index.tolist()
        indices_L = ppdb.df['ATOM'][ppdb.df["ATOM"]['chain_id'] == "L"].index.tolist()
        is_H_first = indices_H ==list(range(0,len(indices_H)))
        if is_H_first:
            ppdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
        else:
            ppdb.df['ATOM'].loc[ppdb.df["ATOM"]["chain_id"] == "H","atom_number"] = list(range(0,len(indices_H)))
            ppdbH = PandasPdb()
            ppdbH.df['ATOM'] = ppdb.df['ATOM'][ppdb.df["ATOM"]['chain_id'] == 'H']
            ppdb.df['ATOM'].loc[ppdb.df["ATOM"]["chain_id"] == "L","atom_number"] = list(range(len(indices_H),len(indices_H)+len(indices_L)))
            ppdbL = PandasPdb()
            ppdbL.df['ATOM'] = ppdb.df['ATOM'][ppdb.df["ATOM"]['chain_id'] == 'L']
            final_pdb = PandasPdb()
            final_pdb.df['ATOM'] = pd.concat([ppdbH.df['ATOM'],ppdbL.df['ATOM']],axis=0)
            final_pdb.df['ATOM'] = final_pdb.df['ATOM'].sort_values(by=["atom_number"])
            final_pdb.df["ATOM"] = final_pdb.df["ATOM"].reset_index().drop(columns=["index"])
            final_pdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
    return

def reorder_rename_Nb(summary_df,pdb,pdb_src,pdb_dest):
    pdb_fvs = summary_df #.loc[summary_df["pdb"]==pdb]
    H_chains = pdb_fvs.Hchain #.tolist()
    # L_chains = pdb_fvs.Lchain.tolist()
    for j in range(len(H_chains)):
        ppdb = PandasPdb()
        ppdb.read_pdb(pdb_src)
        ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df["ATOM"]['chain_id'] == H_chains[j])]
        ppdb.df['ATOM'].loc[ppdb.df["ATOM"]["chain_id"] == H_chains[j],"chain_id"] = "H"
        indices_H = ppdb.df['ATOM'][ppdb.df["ATOM"]['chain_id'] == "H"].index.tolist()
        is_H_first = indices_H ==list(range(0,len(indices_H)))
        if is_H_first:
            ppdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
        else:
            ppdb.df['ATOM'].loc[ppdb.df["ATOM"]["chain_id"] == "H","atom_number"] = list(range(0,len(indices_H)))
            ppdbH = PandasPdb()
            ppdbH.df['ATOM'] = ppdb.df['ATOM'][ppdb.df["ATOM"]['chain_id'] == 'H']
            ppdbL = PandasPdb()
            final_pdb = PandasPdb()
            final_pdb.df['ATOM'] = ppdbH.df['ATOM']
            final_pdb.df['ATOM'] = final_pdb.df['ATOM'].sort_values(by=["atom_number"])
            final_pdb.df["ATOM"] = final_pdb.df["ATOM"].reset_index().drop(columns=["index"])
            final_pdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
    return

def rename_Ab_AF2(pdb_src,pdb_dest):
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_src)
    atmseq = get_atmseq(pdb_src)
    if len(atmseq) == 2:
        ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {'A': 'H', 'B': 'L'}})
    elif len(atmseq)==3:
        ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {'A': 'H', 'B': 'L','C':'A'}})
    ppdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
    return

def rename_Nb_AF2(pdb_src,pdb_dest):
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_src)
    chainids = ppdb.df['ATOM'].chain_id.unique().tolist()
    atmseq = get_atmseq(pdb_src)
    if len(atmseq) == 2:
        ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {'A': 'H', 'B': 'A'}})
    elif len(atmseq)==1:
        ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {f'{chainids[0]}': 'H'}})
    ppdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
    return


def apply_align(pose):
    scheme_ = pyrosetta.rosetta.protocols.antibody.AntibodyNumberingSchemeEnum.IMGT_Scheme
    converter_ = pyrosetta.rosetta.protocols.antibody.AntibodyNumberingConverterMover()
    pyrosetta.rosetta.protocols.antibody.AntibodyNumberingConverterMover.set_from_scheme(converter_,scheme_)
    converter_.apply(pose)
    return pose

def get_ab_metrics(
    pose_1,
    pose_2,
):

    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    pose_i2 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_2)

    results = pyrosetta.rosetta.protocols.antibody.cdr_backbone_rmsds(
        pose_1,
        pose_2,
        pose_i1,
        pose_i2,
    )

    results_labels = [
        'ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms', 'frl_rms', 'l1_rms',
        'l2_rms', 'l3_rms'
    ]
    results_dict = {}
    for i in range(9):
        results_dict[results_labels[i]] = results[i + 1]

    return results_dict

def melt_pyrose(df):
    df = pd.melt(df,id_vars= ['PDB','Seed','Model'],value_vars=['ocd','frh_rms','h1_rms','h2_rms','h3_rms',\
                            'frl_rms','l1_rms','l2_rms','l3_rms'],var_name='Fv_Region',value_name='RMSD')
    return df

def get_coordinates(atom,df):
    """
    Description:
        Extracts cartesian coordinates from PandasPDB df
    Args:
        df: residue df
        atom: atom name
    Output:
        coords: cartesian coordinates of particular atom, [1,1,3]
    """
    X = df.loc[df['atom_name'] == atom].x_coord.iloc[0]
    Y = df.loc[df['atom_name'] == atom].y_coord.iloc[0]
    Z = df.loc[df['atom_name'] == atom].z_coord.iloc[0]
    coords = torch.tensor([X,Y,Z]).unsqueeze(0).unsqueeze(1).unsqueeze(2)
    return coords

def fill_density_original(coords):
    necessary_atoms = ['N','CA','C','O','CB']
    bb_atoms = ['N','CA','C']
    imputed_full_coords = []
    seq = []
    for chain_type in ['H',"L"]:
        chain_seq = []
        res_len_chain_start = coords.df['ATOM'].loc[coords.df['ATOM']['chain_id'] == chain_type].residue_number.min()
        res_len_chain_end = coords.df['ATOM'].loc[coords.df['ATOM']['chain_id'] == chain_type].residue_number.max()
        for i in range(res_len_chain_start,res_len_chain_end+1):
            residue_ = coords.df['ATOM'].loc[(coords.df['ATOM']['residue_number'] == i) & (coords.df['ATOM']['chain_id'] == chain_type)]
            insertions = residue_.insertion.tolist()
            insertions = list(dict.fromkeys(insertions))
            if not insertions:
                pass
            else:
                for j in insertions:
                    residue_ = coords.df['ATOM'].loc[(coords.df['ATOM']['residue_number'] == i) & (coords.df['ATOM']['chain_id'] == chain_type) & (coords.df['ATOM']['insertion'] == j)]
                    missing_atoms = [x for x in necessary_atoms if x not in residue_.atom_name.tolist()]
                    bb_missing_atoms = [x for x in bb_atoms if x not in residue_.atom_name.tolist()]
                    if not missing_atoms:
                        N = get_coordinates('N',residue_)
                        CA = get_coordinates('CA',residue_)
                        C = get_coordinates('C',residue_)
                        O = get_coordinates('O',residue_)
                        CB = get_coordinates('CB',residue_)
                        res_coords = torch.cat([N,CA,C,O,CB],dim=-2)
                        res_coords = res_coords.reshape(1,-1,3)
                        imputed_full_coords.append(res_coords)
                        seq_aa = resindex_to_ressymb[AA[residue_.residue_name.iloc[0]]]
                        chain_seq.append(seq_aa)
                        #print(seq_aa)
                    elif set(missing_atoms)==set(necessary_atoms) or set(missing_atoms) == set(bb_atoms):
                        pass
                    else:
                        pass
                        # res_coords = impute_missing_heavy_atoms(missing_atoms,bb_missing_atoms,residue_)
                        # res_coords = res_coords.reshape(1,-1,3)
                        # imputed_full_coords.append(res_coords)
                        # seq_aa = resindex_to_ressymb[AA[residue_.residue_name.iloc[0]]]
                        # #print(seq_aa)
                        # chain_seq.append(seq_aa)
        seq.append("".join(chain_seq))
    full_coords = torch.cat(imputed_full_coords,dim=0)
    full_seq = ":".join(seq)
    return full_coords, full_seq

def extract_pdb_data(home_dir,pdbfile):
    """
    Processes pdbfiles into getting backbone and sidechain C_beta coordinates, and torsion-relevant coordinates
    """
    explore_ppdb = PandasPdb()
    explore_ppdb.read_pdb(home_dir+pdbfile)
    #Get backbone N, Ca, C, O 
    explore_ppdb_coords = PandasPdb()
    explore_ppdb_coords.df['ATOM'] = explore_ppdb.df['ATOM'].loc[(explore_ppdb.df['ATOM']['atom_name']=='N') |
                                                    (explore_ppdb.df['ATOM']['atom_name']=='CA') |
                                                    (explore_ppdb.df['ATOM']['atom_name']=='C') |
                                                    (explore_ppdb.df['ATOM']['atom_name']=='O') |
                                                    (explore_ppdb.df['ATOM']['atom_name']=='CB')]
    coords,full_seq = fill_density_original(explore_ppdb_coords)
    
    return coords, full_seq

def reorder(pdb_src,pdb_dest):
    ppdb = PandasPdb()
    ppdb.read_pdb(pdb_src)
    atmseq = get_atmseq(pdb_src)
    # print(atmseq)
    for i in atmseq:
        if i =='H':
            H_pdb = PandasPdb()
            H_pdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df["ATOM"]['chain_id'] == 'H')]
            indices_H = H_pdb.df['ATOM'].index.tolist()
            is_H_first = indices_H ==list(range(0,len(indices_H)))
            if is_H_first:
                pass
            else:
                H_pdb.df['ATOM'].loc[H_pdb.df["ATOM"]["chain_id"] == "H","atom_number"] = list(range(1,len(indices_H)+1))
                H_pdb.df['ATOM'].loc[H_pdb.df["ATOM"]["chain_id"] == "H","line_idx"] = list(range(1,len(indices_H)+1))
            if len(atmseq) == 1:
                antigen = None
            # print('H done')
        elif i == 'L':
            L_pdb = PandasPdb()
            L_pdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df["ATOM"]['chain_id'] == 'L')]
            indices_L = L_pdb.df['ATOM'].index.tolist()
            # print('L done')
            if len(atmseq) == 2:
                antigen = None
                # print('defined no antigen!')
        else:
            antigen_id = i
            A_pdb = PandasPdb()
            A_pdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df["ATOM"]['chain_id'] == antigen_id)]
            indices_A = A_pdb.df['ATOM'].index.tolist()
            antigen = antigen_id
            # print('found antigen')

    # print(indices_L)
    
    if 'L' in atmseq:
        L_pdb.df['ATOM'].loc[L_pdb.df["ATOM"]["chain_id"] == "L","atom_number"] = list(range(1+len(indices_H),1+len(indices_H)+len(indices_L)))
        L_pdb.df['ATOM'].loc[L_pdb.df["ATOM"]["chain_id"] == "L","line_idx"] = list(range(1+len(indices_H),1+len(indices_H)+len(indices_L)))
        if antigen!=None:
            A_pdb.df['ATOM'].loc[A_pdb.df["ATOM"]["chain_id"] == antigen_id,"atom_number"] = list(range(1+len(indices_H)+len(indices_L),1+len(indices_H)+len(indices_L)+len(indices_A)))
            A_pdb.df['ATOM'].loc[A_pdb.df["ATOM"]["chain_id"] == antigen_id,"line_idx"] = list(range(1+len(indices_H)+len(indices_L),1+len(indices_H)+len(indices_L)+len(indices_A)))
    elif ('L' not in atmseq ) and (len(atmseq) == 2):
            A_pdb.df['ATOM'].loc[A_pdb.df["ATOM"]["chain_id"] == antigen_id,"atom_number"] = list(range(1+len(indices_H),1+len(indices_H)+len(indices_A)))
            A_pdb.df['ATOM'].loc[A_pdb.df["ATOM"]["chain_id"] == antigen_id,"line_idx"] = list(range(1+len(indices_H),1+len(indices_H)+len(indices_A)))
    final_pdb = PandasPdb()
    final_pdb.df['ATOM'] = H_pdb.df['ATOM']
    
    if 'L' in atmseq:

        final_pdb.df['ATOM'] = pd.concat([final_pdb.df['ATOM'],L_pdb.df['ATOM']],axis=0)

    if antigen!=None:

        final_pdb.df['ATOM'] = pd.concat([final_pdb.df['ATOM'],A_pdb.df['ATOM']],axis=0)

    final_pdb.df['ATOM'] = final_pdb.df['ATOM'].sort_values(by=["line_idx"])
    # print("pdb sorted by atom number")
    final_pdb.df["ATOM"] = final_pdb.df["ATOM"].reset_index().drop(columns=["index"])
    # pdb_df = final_pdb.df["ATOM"]
    # print(f"pdb index reset! Here's proof: {pdb_df.head(5)}")

    final_pdb.to_pdb(path=pdb_dest,records=['ATOM'],gz=False,append_newline=True)
    
    return final_pdb

def melt_af2(df):
    df = pd.melt(df,id_vars= ['PDB','Bound_Unbound','Protein_type'],value_vars=['ocd','frh_rms','h1_rms','h2_rms','h3_rms',\
                            'frl_rms','l1_rms','l2_rms','l3_rms'],var_name='Fv_Region',value_name='RMSD')
    return df

def rename_reorder_native(pdbfile):
    ppdb = PandasPdb().read_pdb(pdbfile)
    chain_ids = ppdb.df['ATOM']['chain_id'].unique().tolist()
    sequence = ppdb.amino3to1()
    for i in chain_ids:
        chain_seq = ''.join(sequence.loc[sequence['chain_id'] == i,'residue_name'])
        try:
            seq = Chain(chain_seq,scheme='chothia')
            chain_type = seq.chain_type
            print(f"This is the chain type determined: {chain_type}")
            if chain_type in ['K','L']:
                chain_type = 'L'
        except Exception as e:
            print(e)
            # elif chain_type not in ['H','K','L']:
            chain_type = i
        ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {i: chain_type}})
    chain_ids = ppdb.df['ATOM']['chain_id'].unique().tolist()
    if chain_ids.count('H') > 1:
        print("More than one H! Not saving ruined structure")
        return pdbfile
    elif chain_ids.count('L') > 1:
        print("More than one L! Not saving ruined structure")
        return pdbfile
    else:
        pdb_src = '/'.join(pdbfile.split("/")[:6])+'/renamed/'+'/'.join(pdbfile.split("/")[6:])
        print(pdb_src)
        pdb_dest = '/'.join(pdbfile.split("/")[:6])+'/renamed/reordered/'+'/'.join(pdbfile.split("/")[6:])
        print(pdb_dest)
        ppdb.to_pdb(path=pdb_src,records=['ATOM'],gz=False,append_newline=True)
        _ = reorder(pdb_src,pdb_dest)
        return True

def rename_native_old_bad(dir_,pdbfile,df,pdb_dest):
    short_name = pdbfile
    print(short_name)
    ppdb = PandasPdb()
    try:
        dir_ = dir_.item()
    except:
        pass
    if os.path.isfile(dir_+pdbfile):
        ppdb.read_pdb(dir_+pdbfile)
        chain_ids = ppdb.df['ATOM']['chain_id'].unique().tolist()
        sequence = ppdb.amino3to1()
        # print(df[(df["PDB"]==short_name)].HeavyChain)
        H_seq = df[(df["PDB"]==short_name)].HeavyChain.item()
        # print(H_seq)
        Hchain = Chain(H_seq, scheme='chothia')
        H_loops = [Hchain.cdr1_seq, Hchain.cdr2_seq, Hchain.cdr3_seq]
        if 'LightChain' not in df.columns:
            L_seq = None
        else:
            L_seq = df[(df["PDB"]==short_name)].LightChain.item()
            Lchain = Chain(L_seq, scheme='chothia')
            L_loops = [Lchain.cdr1_seq, Lchain.cdr2_seq, Lchain.cdr3_seq]

        chain_order = []
        real_chain_ids =[]
        for i in chain_ids:
            # print(i)
            # print(L_seq)
            chain_seq = ''.join(sequence.loc[sequence['chain_id'] == i,'residue_name'])
            # print(chain_seq)
            if chain_seq[1:-1] in H_seq:
                # print("H matching: ")
                # print("from df: {}".format(H_seq))
                # print("from pdb: {}".format(chain_seq))
                chain_order.append('H')
                real_chain_ids.append(i)
                Hseq = chain_seq
            elif (L_seq != None):
                if(chain_seq[1:-1] in L_seq):
                    # print("L matching: ")
                    # print("from df: {}".format(L_seq))
                    # print("from pdb: {}".format(chain_seq))
                    chain_order.append('L')
                    real_chain_ids.append(i)
                    Lseq = chain_seq
                else: 
                    pass
            else:
                pass
        chain_dict= dict(zip(chain_order,real_chain_ids))
        atmseq = get_atmseq(dir_+pdbfile)
        if 'LightChain' not in df.columns:
            if len(atmseq) == 2:
                ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {chain_dict['H']: 'H'}})
            else:
                ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {chain_dict['H']: 'H'}})
        else:
            if len(atmseq) == 2:
                ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {chain_dict['H']: 'H',chain_dict['L']: 'L'}})
            else:
                ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {chain_dict['H']: 'H',chain_dict['L']: 'L'}}) 
        ppdb.to_pdb(path=pdb_dest+pdbfile, records=['ATOM'], gz=False, append_newline=True)

    
def rename_native(pdbs_to_fixdf,i,updated_sabdab_summary):
    native_pdb_info = pdbs_to_fixdf.iloc[i]
    native_dir = native_pdb_info.Dir
    if isinstance(native_dir, str):
        pass
    else:
        native_dir = native_dir.item()
    
    native_pdb = native_pdb_info.PDB #.item()
    if isinstance(native_pdb, str):
        pass
    else:
        native_pdb = native_pdb.item()
    pdbfile = native_dir+native_pdb
    ppdb = PandasPdb().read_pdb(pdbfile)
    native_info = native_dir.split("/")[6].split("_")
    b_ub = native_info[0]
    prot_type = native_info[1]
    print(prot_type)
    pdb_info = updated_sabdab_summary[updated_sabdab_summary['pdb']==native_pdb.split('_')[0]]
    # print(pdb_info)
    chain_id_set = []
    if (b_ub == 'bound') :
        if (prot_type == 'nanobody'):
            for k in range(pdb_info.shape[0]):
                chain_id_set.append([pdb_info.Hchain.tolist()[k],pdb_info.antigen_chain.tolist()[k]])
            atmseq = get_atmseq(native_dir+native_pdb)
            chain_ids_infile = [i for i in atmseq]
            correct_labels = chain_id_set[[True if set(chain_id_set[x]) == set(chain_ids_infile) else False for x in range(len(chain_id_set)) ].index(True)]
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[0]: 'H'}})
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[1]: 'A'}})
        else:
            for k in range(pdb_info.shape[0]):
                chain_id_set.append([pdb_info.Hchain.tolist()[k],pdb_info.Lchain.tolist()[k],pdb_info.antigen_chain.tolist()[k]])
            atmseq = get_atmseq(native_dir+native_pdb)
            chain_ids_infile = [i for i in atmseq]
            correct_labels = chain_id_set[[True if set(chain_id_set[x]) == set(chain_ids_infile) else False for x in range(len(chain_id_set)) ].index(True)]
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[0]: 'H'}})
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[1]: 'L'}})
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[2]: 'A'}})
    elif b_ub == 'unbound':
        if (prot_type == 'nanobody'):
            for k in range(pdb_info.shape[0]):
                chain_id_set.append([pdb_info.Hchain.tolist()[k]])
                
            atmseq = get_atmseq(native_dir+native_pdb)
            chain_ids_infile = [i for i in atmseq]
            correct_labels = chain_id_set[[True if set(chain_id_set[x]) == set(chain_ids_infile) else False for x in range(len(chain_id_set)) ].index(True)]
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[0]: 'H'}})
        else:
            for k in range(pdb_info.shape[0]):
                chain_id_set.append([pdb_info.Hchain.tolist()[k],pdb_info.Lchain.tolist()[k]])
            atmseq = get_atmseq(native_dir+native_pdb)
            chain_ids_infile = [i for i in atmseq]
            correct_labels = chain_id_set[[True if set(chain_id_set[x]) == set(chain_ids_infile) else False for x in range(len(chain_id_set)) ].index(True)]
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[0]: 'H'}})
            ppdb.df['ATOM'] = ppdb.df['ATOM'].replace({'chain_id': {correct_labels[1]: 'L'}})
    print(correct_labels)
    pdb_src = '/'.join(pdbfile.split("/")[:6])+'/renamed/'+'/'.join(pdbfile.split("/")[6:])
    pdb_dest = '/'.join(pdbfile.split("/")[:6])+'/renamed/reordered/'+'/'.join(pdbfile.split("/")[6:])
    print(pdb_dest)
    ppdb.to_pdb(path=pdb_src,records=['ATOM'],gz=False,append_newline=True)
    _ = reorder(pdb_src,pdb_dest)
    return True

def scratch_CDRH3_RMSD(pose_1, pose_2):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    pose_i2 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_2)

    cdr_h3_i_first = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_1)
    cdr_h3_j_first = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i2,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_2)

    cdr_h3_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_1)
    cdr_h3_j_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i2,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_2)
    cdr_h3_i_length = cdr_h3_i_last-cdr_h3_i_first
    cdr_h3_j_length = cdr_h3_j_last-cdr_h3_j_first
    if cdr_h3_i_length != cdr_h3_j_length:
        return "length mismatch"
    else:
        # print("pdb 1 cdr start {}, end {}".format(cdr_h3_i_first,cdr_h3_i_last))
        # print("pdb 2 cdr start {}, end {}".format(cdr_h3_j_first,cdr_h3_j_last))

        pdb_1_cdrh3seq = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_sequence_with_stem(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_1)
        pdb_2_cdrh3seq = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_sequence_with_stem(pose_i2,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_2)
        # print("cdr h3 seq of pdb 1: {}".format(pdb_1_cdrh3seq))
        # print("cdr h3 seq of pdb 2: {}".format(pdb_2_cdrh3seq))

        cdr_H3_i = Pose()
        cdr_H3_j = Pose()
        def subpose_maker(start_,end_,new_pose,src_pose):
            index_vector = pyrosetta.rosetta.utility.vector1_unsigned_long()
            cdr_range= list(range(start_,end_))
            index_vector.extend(cdr_range)
            rosetta.core.pose.pdbslice(new_pose,src_pose,index_vector)
            return new_pose
        cdr_H3_i = subpose_maker(cdr_h3_i_first,cdr_h3_i_last,cdr_H3_i,pose_1)
        #pymover.apply(cdr_H3_i)
        cdr_H3_j = subpose_maker(cdr_h3_j_first,cdr_h3_j_last,cdr_H3_j,pose_2)
        #pymover.apply(cdr_H3_j)
        #CDR_H3_RMSD = pyrosetta.rosetta.core.scoring.CA_rmsd(cdr_H3_i,cdr_H3_j)
        CDR_H3_RMSD = pyrosetta.rosetta.core.scoring.CA_or_equiv_rmsd(cdr_H3_i,cdr_H3_j)
        
        return CDR_H3_RMSD
    
def CDRH3_Bfactors(pose):    

    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)
    cdr_h3_i_first = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose)

    cdr_h3_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose)
    h3_plddt_array = [pose.pdb_info().temperature(i,1) for i in range (cdr_h3_i_first+1, cdr_h3_i_last+1)]  
    return h3_plddt_array

def CDRL3_Bfactors(pose):    

    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)
    cdr_l3_i_first = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l3,pose)

    cdr_l3_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l3,pose)
    l3_plddt_array = [pose.pdb_info().temperature(i,1) for i in range (cdr_l3_i_first+1, cdr_l3_i_last+1)]  
    return l3_plddt_array

# def CDRH3_Bfactors(pose): #Bad function.
#     pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose)

#     cdr_h3_i_first = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose)

#     cdr_h3_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose)
    
#     # cdr_h3_i_length = cdr_h3_i_last-cdr_h3_i_first
    
#     cdr_H3_i = Pose()
    
#     def subpose_maker(start_,end_,new_pose,src_pose):
#         index_vector = pyrosetta.rosetta.utility.vector1_unsigned_long()
#         cdr_range= list(range(start_,end_))
#         index_vector.extend(cdr_range)
#         rosetta.core.pose.pdbslice(new_pose,src_pose,index_vector)
#         return new_pose
#     cdr_H3_i = subpose_maker(cdr_h3_i_first,cdr_h3_i_last,cdr_H3_i,pose)
#     #pymover.apply(cdr_H3_i)
#     #pymover.apply(cdr_H3_j)
#     #CDR_H3_RMSD = pyrosetta.rosetta.core.scoring.CA_rmsd(cdr_H3_i,cdr_H3_j)
#     # CDR_H3_RMSD = pyrosetta.rosetta.core.scoring.CA_or_equiv_rmsd(cdr_H3_i,cdr_H3_j)
#     h3_plddt_array = [cdr_H3_i.pdb_info().temperature(i,1) for i in range (1, cdr_H3_i.total_residue()+ 1)]
        
#     return h3_plddt_array

def var_frag_cropping(pose_1,path_1, pose_2,path_2):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    ag_present = pose_i1.antigen_present()
    # print(f"antigen present: {ag_present}")
    pose_i2 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_2)

    

    h_i_first = pose_1.pdb_info().pdb2pose("H", 1) #pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_1)
    l_i_first = pose_1.pdb_info().pdb2pose("L", 1)
    h_j_first = pose_2.pdb_info().pdb2pose("H", 1) #pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_1)
    l_j_first = pose_2.pdb_info().pdb2pose("L", 1)



    cdr_h4_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h4,pose_1)
    cdr_l4_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l4,pose_1)
    cdr_h4_j_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i2,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h4,pose_2)
    cdr_l4_j_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i2,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l4,pose_2)
    if ag_present:
        i_ag_chainid = pose_i1.get_antigen_chain_string()
        j_ag_chainid = pose_i2.get_antigen_chain_string()
        seqs_i =  get_atmseq(path_1)
        ag_i =seqs_i[i_ag_chainid]
        seqs_j =  get_atmseq(path_2)
        ag_j =seqs_j[j_ag_chainid]
        a_i_first = pose_1.pdb_info().pdb2pose(i_ag_chainid,1)
        # a_i_first = pose_1.pdb_info().pdb2pose(i_ag_chainid,int(pose_1.pdb_info().pose2pdb(len(h_i)+len(l_i)).split(" ")[0]))+1 #pose_1.pdb_info().pdb2pose(i_ag_chainid, 1)
        a_i_last = a_i_first+len(ag_i)-1
        # print(f"first ag number of pred pose: {a_i_first}, last ag num: {a_i_last}")
        # a_j_first = pose_2.pdb_info().pdb2pose(j_ag_chainid,int(pose_2.pdb_info().pose2pdb(len(h_j)+len(l_j)).split(" ")[0]))+1 #pose_2.pdb_info().pdb2pose(j_ag_chainid, 1)
        a_j_first = int(pose_2.total_residue())-len(ag_j)+1 #.pdb_info().pdb2pose(j_ag_chainid,1)
        a_j_last = a_j_first+len(ag_j)-1
        # print(int(pose_2.pdb_info().pose2pdb(len(h_j)+len(l_j)).split(" ")[0]))
        # print(f"first ag number of native pose: {a_j_first}, last ag num: {a_j_last}")

    i_fv = Pose()
    j_fv = Pose()
    def subpose_maker(h_range,l_range,new_pose,src_pose,ag_present,ag_range=Optional):
        index_vector = pyrosetta.rosetta.utility.vector1_unsigned_long()
        all_seq_list=[]
        h_range= list(range(h_range[0],h_range[1]))
        l_range= list(range(l_range[0],l_range[1]))
        all_seq_list.extend(h_range+l_range)
        if ag_present:
            ag_range = list(range(ag_range[0],ag_range[1]+1))
            all_seq_list.extend(ag_range)
        all_seq_list = pd.Series(all_seq_list).unique().tolist()
        index_vector.extend(all_seq_list)
        rosetta.core.pose.pdbslice(new_pose,src_pose,index_vector)
        return new_pose
    if ag_present:
        i_fv = subpose_maker((h_i_first,cdr_h4_i_last+47),(l_i_first,cdr_l4_i_last+47),i_fv,pose_1,ag_present,(a_i_first,a_i_last))
        pred_path = "/".join(path_1.split('/')[:-1])+"/fv_"+"_".join(path_1.split('/')[-1].split("_")[1:])
        i_fv.dump_pdb(pred_path)
        j_fv = subpose_maker((h_j_first,cdr_h4_j_last+47),(l_j_first,cdr_l4_j_last+47),j_fv,pose_2,ag_present,(a_j_first,a_j_last))
        native_crop_path = "/".join(path_2.split("/")[:-1])+"/fv_"+path_2.split("/")[-1]
        j_fv.dump_pdb(native_crop_path)
    else:
        i_fv = subpose_maker((h_i_first,cdr_h4_i_last+47),(l_i_first,cdr_l4_i_last+47),i_fv,pose_1,ag_present)
        pred_path = "/".join(path_1.split('/')[:-1])+"/fv_"+"_".join(path_1.split('/')[-1].split("_")[1:])
        i_fv.dump_pdb(pred_path)
        j_fv = subpose_maker((h_j_first,cdr_h4_j_last+47),(l_j_first,cdr_l4_j_last+47),j_fv,pose_2,ag_present)
        native_crop_path = "/".join(path_2.split("/")[:-1])+"/fv_"+path_2.split("/")[-1]
        j_fv.dump_pdb(native_crop_path)
    return pred_path, native_crop_path

def native_ch1_del(pose_1,path_1):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    ppdb = PandasPdb()
    ppdb.read_pdb(path_1)

    h_i_last = ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id']=='H')].residue_number.unique().tolist()[-1]    
    l_i_last = pose_1.pdb_info().pdb2pose("L",ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id']=='L')].residue_number.unique().tolist()[-1])

    cdr_h4_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h4,pose_1)
    cdr_l4_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l4,pose_1)

    def ch1_deleter(pose,L_range=Optional,H_range=Optional):
        if L_range !=None:
            pyrosetta.rosetta.protocols.grafting.delete_region(pose,L_range[0],L_range[1])
        if H_range!= None:
            pyrosetta.rosetta.protocols.grafting.delete_region(pose,H_range[0],H_range[1])
        return pose
    
    i_L_first = cdr_l4_i_last+47
    i_H_first = cdr_h4_i_last+47
    if l_i_last < i_L_first:
        L_already_cropped = True
    else:
        L_already_cropped = False

    if h_i_last < i_H_first:
        H_already_cropped = True
    else:
        H_already_cropped = False
    pred_path = "/".join(path_1.split('/')[:-1])+"/fv_"+path_1.split('/')[-1]
    if L_already_cropped and H_already_cropped:
        pose_1.dump_pdb(pred_path)
    else:
        if L_already_cropped and H_already_cropped == False:
            H_range = [i_H_first,h_i_last]
            pose_1 = ch1_deleter(pose_1,H_range=H_range)
            pose_1.dump_pdb(pred_path)
        elif L_already_cropped == False and H_already_cropped:
            L_range = [i_L_first,l_i_last]
            pose_1 = ch1_deleter(pose_1,L_range=L_range)
            pose_1.dump_pdb(pred_path)
        elif L_already_cropped == False and H_already_cropped == False:
            H_range = [i_H_first,h_i_last]
            L_range = [i_L_first,l_i_last]
            pose_1 = ch1_deleter(pose_1,H_range=H_range,L_range=L_range)
            pose_1.dump_pdb(pred_path)
    return pred_path

def single_file_var_frag_cropping(pose_1,path_1):
    pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
    ag_present = pose_i1.antigen_present()
    print(f"antigen present: {ag_present}")

    

    h_i_first = pose_1.pdb_info().pdb2pose("H", 1) #pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_start(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h3,pose_1)
    l_i_first = pose_1.pdb_info().pdb2pose("L", 1)



    cdr_h4_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.h4,pose_1)
    cdr_l4_i_last = pyrosetta.rosetta.protocols.antibody.AntibodyInfo.get_CDR_end(pose_i1,pyrosetta.rosetta.protocols.antibody.CDRNameEnum.l4,pose_1)
    if ag_present:
        i_ag_chainid = pose_i1.get_antigen_chain_string()
        # print(i_ag_chainid)
        seqs_i =  get_atmseq(path_1)
        ag_i =seqs_i[i_ag_chainid]
        
        a_i_first = pose_1.pdb_info().pdb2pose(i_ag_chainid,1)
        # a_i_first = pose_1.pdb_info().pdb2pose(i_ag_chainid,int(pose_1.pdb_info().pose2pdb(len(h_i)+len(l_i)).split(" ")[0]))+1 #pose_1.pdb_info().pdb2pose(i_ag_chainid, 1)
        a_i_last = a_i_first+len(ag_i)-1

    i_fv = Pose()
    def subpose_maker(h_range,l_range,new_pose,src_pose,ag_present,ag_range=Optional):
        index_vector = pyrosetta.rosetta.utility.vector1_unsigned_long()
        all_seq_list=[]
        h_range= list(range(h_range[0],h_range[1]))
        l_range= list(range(l_range[0],l_range[1]))
        all_seq_list.extend(h_range+l_range)
        if ag_present:
            ag_range = list(range(ag_range[0],ag_range[1]+1))
            all_seq_list.extend(ag_range)
        all_seq_list = pd.Series(all_seq_list).unique().tolist()
        index_vector.extend(all_seq_list)
        print(index_vector)
        rosetta.core.pose.pdbslice(new_pose,src_pose,index_vector)
        return new_pose
    if ag_present:
        i_fv = subpose_maker((h_i_first,cdr_h4_i_last+47),(l_i_first,cdr_l4_i_last+47),i_fv,pose_1,ag_present,(a_i_first,a_i_last))
        pred_path = "/".join(path_1.split('/')[:-1])+"/fv_"+"_".join(path_1.split('/')[-1].split("_")[1:])
        i_fv.dump_pdb(pred_path)
    else:
        i_fv = subpose_maker((h_i_first,cdr_h4_i_last+47),(l_i_first,cdr_l4_i_last+47),i_fv,pose_1,ag_present)
        pred_path = "/".join(path_1.split('/')[:-1])+"/fv_"+"_".join(path_1.split('/')[-1].split("_")[1:])
        i_fv.dump_pdb(pred_path)
    return pred_path


import pandas as pd
import numpy as np
import argparse
from Bio.PDB import PDBParser
from Bio.PDB import PDBIO
from Bio.PDB.Superimposer import Superimposer
import urllib.request

def get_interface_residues(pdb_file, partners="H_A", cutoff=10.0):
    """Obtain the interface residues in a particular predicted structure. 
    Args
        pdb_file (str): Path to a PDB file
        partners (str): String annotating the docking partners. Partners are 
                        defined as <partner chain 1>_<partner chain 2>. for e.g.
                        A_B, A_HL, ABC_DE, and so on. Note this is applicable for
                        proteins and peptides with canonical amino acids only.
        cutoff (float): Cut-off to determine the interface residues. Default is 
                        10 Angstorms.
    """
    parser = PDBParser()
    structure = parser.get_structure("model", pdb_file)

    partner1 = partners.split("_")[0]
    partner2 = partners.split("_")[1]

    interface_residues = {'partner1': [], 'partner2': []}

    for chain1 in partner1:
        for chain2 in partner2:
            chain1_residues = set(structure[0][chain1].get_residues())
            chain2_residues = set(structure[0][chain2].get_residues())
            int1_residues = []
            int2_residues = []

            for residue1 in chain1_residues:
                for residue2 in chain2_residues:
                    distance = min(
                        [np.linalg.norm(atom1.get_coord() - atom2.get_coord())
                         for atom1 in residue1.get_atoms()
                         for atom2 in residue2.get_atoms()])

                    if distance <= cutoff:
                        int1_residues.append((chain1,residue1))
                        int2_residues.append((chain2,residue2))

            interface_residues['partner1'].extend(list(set(int1_residues)))
            interface_residues['partner2'].extend(list(set(int2_residues)))

    interface_residues['partner1'] = list(set(interface_residues['partner1']))
    interface_residues['partner2'] = list(set(interface_residues['partner2']))

    return interface_residues

def get_interface_residue_b_factors(pdb_file, partners="H_A", cutoff = 10):
    """
    Returns a dictionary of interface residues and their B-factor values.
    :param pdb_file: The path to the PDB file.
    :param chain1: The ID of the first chain.
    :param chain2: The ID of the second chain.
    :param cutoff_distance: The cutoff distance for identifying interface residues.
    """
    structure = PDBParser().get_structure('pdb', pdb_file)
    model = structure[0]
    b_factors = {}
    
    interface_residues = get_interface_residues(pdb_file, partners, cutoff )
    
    for k, v in interface_residues.items():
        for residue in v:
            for atom in residue[-1]:
                if ( atom.get_name() == 'CA' ):
                    b_factors[(k, residue[0], residue[-1].get_id()[1], atom.get_name())] = atom.get_bfactor()
            
    return b_factors

def renumber_pdb(old_pdb, renum_pdb=None):
    """_summary_

    Args:
        old_pdb (_type_): _description_
        renum_pdb (_type_, optional): _description_. Defaults to None.

    Code by Jeff Ruffolo
    """    
    
    if not exists(renum_pdb):
        renum_pdb = old_pdb

    success = False
    time.sleep(5)
    for i in range(10):
        try:
            with open(old_pdb, 'rb') as f:
                response = requests.post(
                    'http://www.bioinf.org.uk/abs/abnum/abnumpdb.cgi',
                    params={
                        "plain": "1",
                        "output": "-HL",
                        "scheme": "-c"
                    },
                    files={"pdb": f},
                )

            success = response.status_code == 200 and not ("<html>"
                                                           in response.text)

            if success:
                break
            else:
                time.sleep((i + 1) * 5)
        except requests.exceptions.ConnectionError:
            time.sleep(60)

    if success:
        new_pdb_data = response.text
        with open(renum_pdb, "w") as f:
            f.write(new_pdb_data)
    else:
        print(
            "Failed to renumber PDB. This is likely due to a connection error or a timeout with the AbNum server."
        )

# Getting Ab

def extract_Ab(pdbfile,pdb_dest=None):
    print(pdbfile)
    """_summary_

    Args:
        pdbfile (_type_): input pdbfile
        pdb_dest (_type_, optional): output pdb destination. Defaults to None.

    Returns:
        _type_: True if function successful.
    """    
    if pdb_dest != None:
        pass
    else:
        pdb = 'unbound_'+pdbfile.split('/')[-1]
        pdb_dest = '/'.join(pdbfile.split('/')[:-1])+'/'+pdb
        print(pdb_dest)
    try:
        ppdb = PandasPdb().read_pdb(pdbfile)
        unbound_ppdb = PandasPdb()
        unbound_ppdb.df['ATOM'] = ppdb.df['ATOM'][(ppdb.df['ATOM']['chain_id']=='H')|(ppdb.df['ATOM']['chain_id']=='L')]
        # print(unbound_ppdb.df['ATOM'])
        unbound_ppdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
        
    except:
        print('biopandas failed')
        # Confirm that unbound PDB is good
    try:
        pose_1 = pose_from_pdb(pdb_dest)
        pose_i1 = pyrosetta.rosetta.protocols.antibody.AntibodyInfo(pose_1)
        # print(pose_i1)
        # return True
    except:
        print('pyrosetta AntibodyInfo unable to call data')
        # return False'

#I made predicted_EDMGen 
#edmmaker: IgVAE code
def edmmaker(stacked):
    # Input: B x n x 3.
    #stacked, stacked_t=batchtensormaker(home_dir,batchsize)
    #print("stacked size: {}".format(stacked.size()))
    n = stacked.size(1)
    gij = torch.bmm(stacked, stacked.transpose(1, 2))
    gii = torch.diagonal(gij, dim1=-2, dim2=-1)[:, None, :]
    gii = gii.repeat(1, n, 1)
    EDM = gii + gii.transpose(1, 2) - 2*gij
    #output: B x n x n
    EDM = torch.sqrt(torch.abs(EDM))[:, :, :]
    #print("EDM size: {}".format(EDM.size()))

    #max_dist=torch.max(EDM, 0)[0] #.unsqueeze(1,2)
    #print("max_dist size: {}".format(max_dist.size()))
    #EDM=torch.div(EDM,max)
    return EDM

def predicted_EDMGen(pred_coords):
    #pred_coords=pred_coords.reshape(pred_coords.size(0),pred_coords.size(1)*pred_coords.size(2),-1)
    # Input: B x n x 3.
    stacked=pred_coords
    n = stacked.size(1)
    stacked_check=stacked!=stacked
    if 1 in stacked_check:
        print("coordinates fed to this function have a NaN....")
    stackedtranspose=stacked.transpose(1,2)
    stackedT_check=stackedtranspose!=stackedtranspose
    if 1 in stackedT_check:
        print("The coordinate tranpose is producing a NaN.")
    gij = torch.bmm(stacked, stacked.transpose(1, 2))
    GIJ_check = gij!=gij
    if 1 in GIJ_check:
        print("GIJ (torch.bmm) is producing the NaN.")
    gii =  torch.diagonal(gij, dim1=-2, dim2=-1)[:, None, :]
    GII_check = gii!=gii
    if 1 in GII_check:
        print("GII has a NaN.")
    gii = gii.repeat(1, n, 1)
    GII_check2 = gii!=gii
    if 1 in GII_check:
        print("GII (1,n,1) repeat has a NaN.")
    gii_t=gii.transpose(1,2)
    GII_transpose_check = gii_t!=gii_t
    if 1 in GII_transpose_check:
        print("GII.transpose(1,2) has a NaN.")
    neg_twoxgij=-2*gij
    negtwogij_check=neg_twoxgij!=neg_twoxgij
    if 1 in negtwogij_check:
        print("-2*gij has the NaN!")
    EDM = gii + gii_t + neg_twoxgij
    eps=0.00001
    EDM=torch.add(EDM,eps)
    EDM = torch.sqrt(EDM)[:, :, :]
    return EDM

def get_interface_res_fast(pdbfile,partners,cutoff=10.0):
    """Obtain the interface residues in a particular predicted structure. 
    Args
        pdb_file (str): Path to a PDB file
        partners (str): String annotating the docking partners. Partners are 
                        defined as <partner chain 1>_<partner chain 2>. for e.g.
                        A_B, A_HL, ABC_DE, and so on. Note this is applicable for
                        proteins and peptides with canonical amino acids only.
        cutoff (float): Cut-off to determine the interface residues. Default is 
                        10 Angstorms.

    * Importantly: assumes single ligand chain, multiple protein chains!!
    """
    ppdb1 = PandasPdb().read_pdb(pdbfile)
    chain_order= ppdb1.df['ATOM'].chain_id.unique().tolist()
    # total_chains = len(chain_order)
    ligand_chain = partners.split("_")[-1]
    if ligand_chain not in chain_order:
        return None, None
    else:
        ligand_idx = chain_order.index(ligand_chain)
        prot_chains = [x for x in partners.split("_")[0]]
        prot_idx = [chain_order.index(x) for x in prot_chains]
        # print(prot_idx)
        ca_df = ppdb1.df['ATOM'][ppdb1.df['ATOM']['atom_name']=='CA'].reset_index().drop(columns=['index'])
        ca_coords = coord_extractor(ca_df)
        edm = predicted_EDMGen(ca_coords).squeeze()
        # print(edm.shape)
        # print([[0,prot_chains[x]] for x in chain_order])
        res_lens = [[0,ca_df[ca_df['chain_id']==x].shape[0]] for x in chain_order]
        prot_res_raw = []
        ligand_res_raw = []
        ligand_res = []
        prot_res = []
        for idces in prot_idx:
            # print(idces)
            # print(prot_chains[idces])
            ligand_indx_start = np.sum([res_lens[x][1] for x in range(ligand_idx)])
            ligand_indx_end = np.sum([res_lens[x][1] for x in range(ligand_idx+1)])
            if idces == 0:
                prot_indx_start = 0
            else:  
                prot_indx_start = np.sum([res_lens[i][1] for i in range(idces)])
            prot_indx_end = np.sum([res_lens[i][1] for i in range(idces+1)])
            interface_edm = edm[prot_indx_start:prot_indx_end,ligand_indx_start:ligand_indx_end]
            interface_res_prot,interface_res_ligand = torch.where(interface_edm<=cutoff)
            toggled_dists = interface_edm[list(interface_res_prot.numpy()),list(interface_res_ligand.numpy())]
            # print(toggled_dists)
            true_prot_idx = pd.Series(interface_res_prot.numpy() + prot_indx_start).unique().tolist()
            true_lig_idx = pd.Series(interface_res_ligand.numpy() + ligand_indx_start).unique().tolist()
            # prot_plddts_raw = ca_df.loc[true_prot_idx,'b_factor'].tolist()
            ligand_res.extend(set(true_lig_idx))
            prot_res.extend(set(true_prot_idx))
            # print("protein idx: {}".format(ca_df.iloc[list(true_prot_idx)].residue_number.tolist()))
            # print("ligand idx: {}".format(ca_df.iloc[list(true_lig_idx)].residue_number.tolist()))
            prot_res_raw.extend(ca_df.iloc[list(true_prot_idx)].residue_number.tolist())
            ligand_res_raw.extend(ca_df.iloc[list(true_lig_idx)].residue_number.tolist())
        
        prot_plddts = ca_df.loc[prot_res,'b_factor'].tolist()
        prot_chainids = ca_df.loc[prot_res,'chain_id'].tolist()
        prot_info = [(prot_chainids[x],'CA',prot_res_raw[x]) for x in range(len(prot_res_raw))]
        lig_info = [(ligand_chain,'CA',ligand_res_raw[x]) for x in range(len(ligand_res_raw))]
        prot_zip = zip(prot_info,prot_plddts)
        plddts = dict(prot_zip)
        lig_plddts = ca_df.loc[ligand_res,'b_factor'].tolist()
        lig_zip = zip(lig_info,lig_plddts)
        plddts.update(lig_zip)
        i_plddt = np.mean(list(plddts.values()))
        # print(i_plddt)
        return plddts,i_plddt
    
# Stitching unbound renumbered to antigen bound

def renumbered_stitch(bound_file,unbound_renum_file):
    print(bound_file)
    pdb_dest_name = "renum_"+bound_file.split("/")[-1]
    print("/".join(bound_file.split("/")[:-1]))
    pdb_dest = "/".join(bound_file.split("/")[:-1])+'/'+pdb_dest_name
    ppdb1 = PandasPdb().read_pdb(bound_file)
    ppdb2 = PandasPdb().read_pdb(unbound_renum_file)
    antigen_coords = ppdb1.df['ATOM'][ppdb1.df['ATOM']['chain_id']=='A']
    renumbered_coords = ppdb2.df['ATOM'][(ppdb2.df['ATOM']['chain_id']=='H')|(ppdb2.df['ATOM']['chain_id']=='L')]
    stitched_pdb = PandasPdb()
    stitched_pdb.df['ATOM'] = pd.concat([renumbered_coords,antigen_coords],axis=0)
    stitched_pdb.to_pdb(path=pdb_dest, records=None, gz=False, append_newline=True)
    return pdb_dest

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