import torch
from biopandas.pdb import PandasPdb
import torch.nn.functional as F
import pandas



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

def coord_extractor(df,chainids=None):
    if chainids!=None:
        df = df[df['chain_id'].isin(chainids)]
    else:
        pass
    df = df[df['atom_name']=='CA']
    X = df['x_coord'].tolist()
    Y = df['y_coord'].tolist()
    Z = df['z_coord'].tolist()
    coords = torch.tensor([X,Y,Z]).transpose(0,1).unsqueeze(0)
    return coords

def rmsd_(pred,true):
    pred_aligned,R,t = kabsch(pred.type(torch.float64),true.type(torch.float64))
    # print("first 5 true coords: {}".format(true[:,:5,:]))
    # print("first 5 unaligned coords: {}".format(pred[:,:5,:]))
    # print("first 5 aligned coords: {}".format(pred_aligned[:,:5,:]))

    # print("last 5 true coords: {}".format(true[:,-5:,:]))
    # print("last 5 unaligned coords: {}".format(pred[:,-5:,:]))
    # print("last 5 aligned coords: {}".format(pred_aligned[:,-5:,:]))

    rmsd = torch.mean(torch.sqrt(torch.sum((pred_aligned-true).pow(2),-1)),-1)
    return pred_aligned,rmsd, R, t

def pdb_align(pdb1,pdb2,chainids1,chainids2):
    ppdb1 = PandasPdb().read_pdb(pdb1)
    ppdb2 = PandasPdb().read_pdb(pdb2)
    coord1 = coord_extractor(ppdb1.df['ATOM'],chainids=chainids1)
    coord2 = coord_extractor(ppdb2.df['ATOM'],chainids=chainids2)
    coord1_length = coord1.shape[1]
    coord2_length = coord2.shape[1]
    if coord1_length < coord2_length:
        min_length = coord1_length
    elif coord2_length < coord1_length:
        min_length = coord2_length
    else:
        min_length = coord1_length
    
    cropped_coord1 = coord1[:,:min_length,:]
    cropped_coord2 = coord2[:,:min_length,:]
    aligned_struc, rmsd, R, t = rmsd_(cropped_coord2,cropped_coord1)
    return aligned_struc.squeeze(), rmsd, ppdb2, R, t

def align_and_dump_pdb(pdb1,pdb2,chainids1=None,chainids2=None):
    aligned_struc, rmsd, ppdb2, R, t = pdb_align(pdb1,pdb2,chainids1,chainids2)
    ca_ppdb2 = PandasPdb()
    ca_ppdb2.df['ATOM'] = ppdb2.df['ATOM'][ppdb2.df['ATOM']['atom_name']=='CA']
    ca_ppdb2.df['ATOM'].loc[:,'x_coord'] = aligned_struc[:,0].tolist()
    ca_ppdb2.df['ATOM'].loc[:,'y_coord'] = aligned_struc[:,1].tolist()
    ca_ppdb2.df['ATOM'].loc[:,'z_coord'] = aligned_struc[:,2].tolist()
    aligned_dest = "/".join(pdb2.split('/')[:-1])+ "/aligned_"+pdb2.split('/')[-1]
    print(aligned_dest)
    ca_ppdb2.to_pdb(path=aligned_dest,records=['ATOM'])
    return rmsd