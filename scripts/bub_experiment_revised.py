# Imports


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
import warnings
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Sequence, List, Optional, Union
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from Benchmarking.benchmark.ops.all_funcs import * 
from Benchmarking.benchmark.ops.protein import *
from Benchmarking.benchmark.ops.benchmark_clean_funcs import *
import colorcet as cc
from scipy.stats import mannwhitneyu, ranksums
from scipy import stats
import statsmodels #.api as sm
import shutil

# palette = sns.color_palette(['#3187DE','#FF88AC','#02AA9D','#FFDF00'])
# palette = sns.color_palette(['#3187DE','#FFDF00','#02AA9D','#FF88AC'])
palette = sns.color_palette(['#FF88AC','#02AA9D','#FFDF00','#3187DE'])
# pastel_palette = sns.color_palette(['#78b6f5','#f7bacc','#96e0db','#fff18f'])
# pastel_palette = sns.color_palette(['#78b6f5','#fff18f','#96e0db','#f7bacc'])
edgecolor = '0'
pastel_palette = sns.color_palette(['#f7bacc','#96e0db','#fff18f','#78b6f5','#FFC8A2','#c997c6'])
pastel_palette2 = sns.color_palette(['#f7bacc','#96e0db','#FFDF00','#78b6f5','#FFC8A2','#c997c6'])
pastel_palette2
extended_pastel_palette = sns.color_palette(['#f7bacc','#96e0db','#FFDF00','#78b6f5','#f7bacc','#c997c6','#57C288','#D66E5C','#AAD2EC'])
extended_pastel_palette
plt.rcParams['svg.fonttype'] = 'none'

# sns.set_style("darkgrid")
sns.set_style("whitegrid",{"grid.color":"0.6"})
# custom_params = {"axes.spines.right": False, "axes.spines.top": False}
# sns.set_theme(style="ticks", rc=custom_params,font_scale=3)
sns.set_theme(font_scale=1.5)
linewidth = 2
edgecolor = 'black'

#artificmissing1 =pd.read_csv("results/round_7/datafiles/artificial_bound_ab_multiseed_addnl_rmsds.csv",index_col='Unnamed: 0')
#artific_2 = pd.read_csv("results/round_7/datafiles/artificial_bound_ab_multiseed_rmsds.csv",index_col='Unnamed: 0')
#artific_3 = pd.read_csv("results/round_7/datafiles/artificial_bound_nb_multiseed_addnl_rmsds.csv",index_col='Unnamed: 0')
#artific_4 = pd.read_csv("results/round_7/datafiles/artificial_bound_nb_multiseed_rmsds.csv",index_col='Unnamed: 0')

#artific_unbound_nb_rmsds = pd.concat([artificmissing1,artific_2],ignore_index=True)
# artific_nb_rmsds= pd.read_csv("results/round_8/datafiles/artificial_unboundnb_rmsds.csv",index_col='Unnamed: 0').drop(columns=['Pred_Dir','Pred_PDB']).rename(columns={"AF3_Dir":"Pred_Dir","AF3_PDB":"Pred_PDB"})
# artific_nb_rmsds= artific_nb_rmsds[artific_nb_rmsds['Native_PDB'].str.contains('8cy7_2')].reset_index(drop=True)
# datadir = '/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/results/round_7/datafiles/'
# artificial_nb1_rmsds = pd.read_csv(f'{datadir}artificial_bound_nb_multiseed_rmsds.csv',index_col='Unnamed: 0').reset_index(drop=True)
# artificial_nb2_rmsds = pd.read_csv(f'{datadir}artificial_bound_nb_multiseed_addnl_rmsds.csv',index_col='Unnamed: 0').reset_index(drop=True)
# artificial_nb3_rmsds = pd.read_csv(f"{datadir}artificial_bound_nb_multiseed_datastructure_additional_2_rmsds.csv",index_col='Unnamed: 0').reset_index(drop=True)
# artificial_nb_rmsds = pd.concat([artificial_nb1_rmsds,artificial_nb2_rmsds,artificial_nb3_rmsds,artific_nb_rmsds]).drop_duplicates()

# nb_trainnonredund=pd.read_csv("/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/results/round_8/datafiles/train_red_dropped_nb_finalbench.csv").reset_index(drop=True).drop(columns=['index'])
# ab_trainnonredund=pd.read_csv("/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/results/round_8/datafiles/train_red_dropped_ab_finalbench.csv").reset_index(drop=True).drop(columns=['index'])
# vhsim_addedbench  = pd.read_csv("results/round_7/datafiles/complete_af3_benchmark.csv").drop(columns=['Unnamed: 0']).iloc[-60:,:]
# final_af3_benchmark = pd.concat([nb_trainnonredund, ab_trainnonredund,vhsim_addedbench], axis=0).reset_index(drop=True)
# final_af3_benchmark.loc[-60:,['PDB_short']] = ['_'.join(x.split('_')[-2:]) for x in final_af3_benchmark.Native_PDB.tolist()]
# final_af3_benchmark.loc[-60:,['PDB_short']] = final_af3_benchmark['PDB_short'].astype(str)
# final_af3_benchmark['Seed'] = [int([y for y in x.split('seed')[-1].split('_') if y!=''][0]) for x in final_af3_benchmark.AF3_PDB.tolist()]
# final_af3_benchmark['Model'] = [int([y.split('.')[0] for y in x.split('model')[-1].split('_') if y!=''][0]) for x in final_af3_benchmark.AF3_PDB.tolist()]
# ranking_df = pd.read_csv("results/round_6/datafiles/af3_withproperranking_results.csv").drop(columns=['Unnamed: 0'])
# ranking_df = ranking_df.rename(columns={"Dir":"AF3_Dir","PDB":"AF3_PDB"})
# fulldf_fixed = pd.read_csv("results/round_6/datafiles/af3_withauxdata_andproperranking_results_fixeddf.csv").drop(columns=['Unnamed: 0'])

# # antibodies
# ab_sabdab_path = "../sabdab_summary_all_240531.tsv"
# ab_sabdab = pd.read_csv(ab_sabdab_path,delimiter='\t')

# # nanobodies
# nb_sabdab_path = "../20240604_0067950_summary_nanobodies.tsv"
# nb_sabdab = pd.read_csv(nb_sabdab_path,delimiter='\t')

# #Final benchmark has been filtered to only include the PDBs that had antigens of type protein, peptide, or protein | protein
# def final_culled(final_af3_benchmark,pdbslist):
#     pdbs = []
#     for x in pdbslist:
#         pdbs.append(final_af3_benchmark[final_af3_benchmark['PDB_short'].str.contains(x)])
#     return pd.concat(pdbs,ignore_index=True)
# final_ab_benchmark = final_culled(final_af3_benchmark,ab_sabdab[(ab_sabdab['pdb'].isin([x.split('_')[0] for x in final_af3_benchmark.PDB_short.unique().tolist()]))\
#     &(ab_sabdab['antigen_type'].isin(['protein','peptide','protein | protein']))].pdb.unique().tolist())
# final_nb_benchmark = final_culled(final_af3_benchmark,nb_sabdab[(nb_sabdab['pdb'].isin([x.split('_')[0] for x in final_af3_benchmark.PDB_short.unique().tolist()]))\
#     &(nb_sabdab['antigen_type'].isin(['protein','peptide','protein | protein']))].pdb.unique().tolist())
# final_benchmark = pd.concat([final_ab_benchmark,final_nb_benchmark],ignore_index=True)
# final_af3_benchmark = pd.concat([final_af3_benchmark[final_af3_benchmark['Bound_Unbound']=='unbound'],final_benchmark],ignore_index=True)
# bad_pdbs = ['7zxu_1.pdb','8cby_0.pdb','7yiz_1.pdb']
# final_af3_benchmark = final_af3_benchmark[~final_af3_benchmark['PDB_short'].isin(bad_pdbs)]

# boundnbs_list = final_af3_benchmark[(final_af3_benchmark['Bound_Unbound']=='bound')&(final_af3_benchmark['Protein_type']=='nanobody')].PDB_short.unique().tolist()
# boundabs_list = final_af3_benchmark[(final_af3_benchmark['Bound_Unbound']=='bound')&(final_af3_benchmark['Protein_type']=='antibody')].PDB_short.unique().tolist()

# unboundnbs_list = final_af3_benchmark[(final_af3_benchmark['Bound_Unbound']=='unbound')&(final_af3_benchmark['Protein_type']=='nanobody')].PDB_short.unique().tolist()
# unboundabs_list = final_af3_benchmark[(final_af3_benchmark['Bound_Unbound']=='unbound')&(final_af3_benchmark['Protein_type']=='antibody')].PDB_short.unique().tolist()

# artificial_nb_rmsds=pd.read_csv('/home/ftalib1/scr4_jgray21/ftalib1/AF3_Benchmark/New_AF3_Benchmark/results/round_8/datafiles/artificial_unboundnb_rmsds_complete.csv',index_col='Unnamed: 0') #.reset_index(drop=True)
# # artificial_nb_rmsds['PDB_short'] = ['_'.join(x.split('_')[-2:]) for x in artificial_nb_rmsds.Native_PDB.tolist()]
# # artificial_nb_rmsds['Seed'] = [int([y for y in x.split('seed')[-1].split('_') if y!=''][0]) for x in artificial_nb_rmsds.Pred_PDB.tolist()]
# artificial_nb_rmsds = artificial_nb_rmsds[artificial_nb_rmsds['PDB_short'].isin(boundnbs_list)]
# redonb_unbound_df = artificial_nb_rmsds[artificial_nb_rmsds['H3_local_RMSD']=='length mismatch']
# redonb_unbound_df['Native_PDB'] = ['renum_'+x for x in redonb_unbound_df.Native_PDB.tolist()]
# redonb_df=redonb_unbound_df.reset_index(drop=True)
# for i in trange(redonb_df.shape[0]):
#     pose_1 = pose_from_pdb(os.path.join(redonb_df.iloc[i].Native_Dir,redonb_df.iloc[i].Native_PDB))
#     pose_2 = pose_from_pdb(redonb_df.iloc[i].Pred_PDB)
#     localh3 = scratch_CDRH3_RMSD(pose_1,pose_2)
#     results = get_ab_metrics(pose_1,pose_2)
#     redonb_df.loc[i,'H3_local_RMSD'] = localh3
#     for j in list(results.keys()):
#         redonb_df.loc[i,j] = results[j]
# artificial_nb_rmsds.iloc[artificial_nb_rmsds[artificial_nb_rmsds['H3_local_RMSD']=='length mismatch'].index.values,:] = redonb_df

# artific_unbound_ab_rmsds= pd.read_csv("results/round_8/datafiles/artificial_boundab_rmsds.csv",index_col='Unnamed: 0')
# artific_unbound_ab_rmsds['PDB_short'] = ['_'.join(x.split('_')[-2:]) for x in artific_unbound_ab_rmsds.Native_PDB.tolist()]
# artific_unbound_ab_rmsds['Seed'] = [int([y for y in x.split('seed')[-1].split('_') if y!=''][0]) for x in artific_unbound_ab_rmsds.Pred_PDB.tolist()]
# artific_unbound_ab_rmsds = artific_unbound_ab_rmsds[artific_unbound_ab_rmsds['PDB_short'].isin(boundabs_list)]

# real_bound_nbs = pd.read_csv("results/round_8/datafiles/bound_nb_multiseed_rmsds.csv",index_col='Unnamed: 0')
# real_bound_nbs['Seed'] = [int([y for y in x.split('seed')[-1].split('_') if y!=''][0]) for x in real_bound_nbs.AF3_PDB.tolist()]
# real_bound_nbs['PDB_short'] = ['_'.join(x.split('_')[-2:]) for x in real_bound_nbs.Native_PDB.tolist()]
# real_bound_nbs = real_bound_nbs[real_bound_nbs['PDB_short'].isin(boundnbs_list)]
# real_bound_abs = pd.read_csv("results/round_8/datafiles/bound_ab_multiseed_rmsds.csv",index_col='Unnamed: 0')
# real_bound_abs['PDB_short'] = ['_'.join(x.split('_')[-2:]) for x in real_bound_abs.Native_PDB.tolist()]
# real_bound_abs = real_bound_abs[real_bound_abs['PDB_short'].isin(boundabs_list)]

# real_unbound_nbs = pd.read_csv("results/round_8/datafiles/unbound_nb_multiseed_rmsds.csv",index_col='Unnamed: 0')
# real_unbound_nbs['PDB_short'] = ['_'.join(x.split('_')[-2:]) for x in real_unbound_nbs.Native_PDB.tolist()]
# real_unbound_nbs = real_unbound_nbs[real_unbound_nbs['PDB_short'].isin(unboundnbs_list)]
# real_unbound_abs = pd.read_csv("results/round_8/datafiles/unbound_ab_multiseed_rmsds.csv",index_col='Unnamed: 0')
# real_unbound_abs['PDB_short'] = ['_'.join(x.split('_')[-2:]) for x in real_unbound_abs.Native_PDB.tolist()]
# real_unbound_abs = real_unbound_abs[real_unbound_abs['PDB_short'].isin(unboundabs_list)]

# def get_20seeds(df):
#        pdb_colname = [x for x in df.columns if 'PDB' in x and 'Native' not in x][0]
#        df[pdb_colname] = [x.split('/')[-1] for x in df[pdb_colname].tolist()]
#        # df['PDB_short'] = ['_'.join(x.split('_seed')[0].split('_')[-2:])+'.pdb' for x in df[pdb_colname].tolist()]
#        df['PDB_short'] = ['_'.join(x.split('_seed')[0].split('_')[-2:]) for x in df['Native_PDB'].tolist()]
#        pdblist = df.PDB_short.unique().tolist()
#        twentyseed_list = []
#        for i in pdblist:
#               pdb_row = df[df['PDB_short']==i]
#               seeds = pdb_row.Seed.unique().tolist()[:20]
#               if len(seeds)<20:
#                      print(i)
#               pdb_row = pdb_row[pdb_row['Seed'].isin(seeds)].drop_duplicates(subset=['Seed'])
#               twentyseed_list.append(pdb_row)
#        return pd.concat(twentyseed_list)

# artific_unbound_nb_rmsds = get_20seeds(artificial_nb_rmsds)
# artific_unbound_ab_rmsds = get_20seeds(artific_unbound_ab_rmsds)
# real_bound_nbs = get_20seeds(real_bound_nbs)
# real_bound_abs = get_20seeds(real_bound_abs)

# real_unbound_nbs = get_20seeds(real_unbound_nbs)
# real_unbound_abs = get_20seeds(real_unbound_abs)

# artific_df = pd.concat([artific_unbound_nb_rmsds,artific_unbound_ab_rmsds,real_bound_nbs,real_bound_abs,real_unbound_nbs,real_unbound_abs],keys=['artificial unbound nb','artificial unbound ab','bound nb','bound ab','unbound nb','unbound ab']).reset_index().drop(columns='level_1').rename(columns={'level_0':'Real versus Artificial'}) #.to_csv("results/round_9/datafiles/bubexperiment_fulldata.csv")
#print(artific_df)

def annot_group():

    return

artific_df = pd.read_csv('results/round_9/datafiles/bubexperiment_fulldata.csv',index_col='Unnamed: 0')
artific_df['Real versus Artificial'] = [x.split(' ')[0] for x in artific_df['Real versus Artificial'].tolist()]

melted_artific_df= pd.melt(artific_df[artific_df['H3_local_RMSD']!='length mismatch'],id_vars = ['Protein_type', 'Bound_Unbound','Real versus Artificial',
       'Seed', 'PDB_short'],value_vars=['ocd', 'frh_rms', 'h1_rms', 'h2_rms', 'h3_rms','H3_local_RMSD','frl_rms',
       'l1_rms', 'l2_rms', 'l3_rms'],var_name = "Fv_Region",value_name = "RMSD (Å)")
melted_artific_df['RMSD (Å)']=melted_artific_df['RMSD (Å)'].astype(float) 
melted_artific_df['Fv_Region']=melted_artific_df['Fv_Region'].astype(str) 
#melted_artific_df
# Bound-Unbound with enough seeds experiment

sns.set_style("whitegrid",{"grid.color":"0.6"})
g = sns.catplot(data=melted_artific_df[melted_artific_df['Fv_Region'].isin(['H3_local_RMSD','h3_rms'])].sort_values(by=['Bound_Unbound'],ascending=False),\
                 x='Fv_Region',y='RMSD (Å)',hue='Real versus Artificial',hue_order=['unbound','bound','artificial']\
                     ,col='Protein_type',col_order=['nanobody','antibody'],palette=extended_pastel_palette,linewidth=linewidth,legend=True,kind='boxen')
# g.ylim=((0,10))
g.set(ylim=(0, 10),ylabel='CDR H3 global RMSD (Å)')
# # g.set(ylim=g.axes.get_ylim()[::-1])
g.refline(y=1.0,ls='--',color='#222222')
g.set_titles(col_template="{col_name}")
g.set(xlabel=None)
sns.set(font_scale=1.35)
plt.savefig('results/round_9/plots/artificialtest_fixednaming.png')
plt.savefig('results/round_9/plots/artificialtest_fixednaming.svg',format='svg',dpi=500,transparent=True,bbox_inches='tight')

ptypes = ['ab','nb']
rmsdtypes = ['h3_rms','H3_local_RMSD']
for ptype in ptypes:
       for rmsdtype in rmsdtypes:
              print(f'ptype: {ptype}, rmsdtype: {rmsdtype}')
              group1 = melted_artific_df[(melted_artific_df['Protein_type']==ptype)&(melted_artific_df['Real versus Artificial']==f'unbound')&(melted_artific_df['Fv_Region']==rmsdtype)]['RMSD (Å)'].dropna()
              print(f'len of unbound {ptype} :{len(group1.tolist())}')
              print(f'describing group 1: {group1.describe()}')
              group2 = melted_artific_df[(melted_artific_df['Protein_type']==ptype)&(melted_artific_df['Real versus Artificial']==f'artificial')&(melted_artific_df['Fv_Region']==rmsdtype)]['RMSD (Å)'].dropna()
              print(f'describing group 2: {group2.describe()}')
              print(f'len of artificial unbound {ptype} :{len(group2.tolist())}')

              bub_compare = mannwhitneyu(group1.astype(float).tolist(),group2.astype(float).tolist())
              print(f"{ptype} p-val for {rmsdtype} unbound v artif. unbound: {bub_compare[1]}")

              group1 = melted_artific_df[(melted_artific_df['Protein_type']==ptype)&(melted_artific_df['Real versus Artificial']==f'unbound')&(melted_artific_df['Fv_Region']==rmsdtype)]['RMSD (Å)'].dropna()
              group2 = melted_artific_df[(melted_artific_df['Protein_type']==ptype)&(melted_artific_df['Real versus Artificial']==f'bound')&(melted_artific_df['Fv_Region']==rmsdtype)]['RMSD (Å)'].dropna()
              print(f'describing group 1: {group1.describe()}')
              print(f'describing group 2: {group2.describe()}')
              print(f'len of unbound {ptype} :{len(group1.tolist())}')
              print(f'len of bound {ptype} :{len(group2.tolist())}')

              bub_compare = mannwhitneyu(group1.astype(float).tolist(),group2.astype(float).tolist())
              print(f"{ptype} p-val for {rmsdtype} unbound v bound: {bub_compare[1]}")

              group1 = melted_artific_df[(melted_artific_df['Protein_type']==ptype)&(melted_artific_df['Real versus Artificial']==f'bound')&(melted_artific_df['Fv_Region']==rmsdtype)]['RMSD (Å)'].dropna()
              group2 = melted_artific_df[(melted_artific_df['Protein_type']==ptype)&(melted_artific_df['Real versus Artificial']==f'artificial')&(melted_artific_df['Fv_Region']==rmsdtype)]['RMSD (Å)'].dropna()
              print(f'describing group 1: {group1.describe()}')
              print(f'describing group 2: {group2.describe()}')
              print(f'len of bound {ptype} :{len(group1.tolist())}')
              print(f'len of artificial unbound {ptype} :{len(group2.tolist())}')

              bub_compare = mannwhitneyu(group1.astype(float).tolist(),group2.astype(float).tolist())
              print(f"{ptype} p-val for {rmsdtype} bound v artific. unbound: {bub_compare[1]}")
