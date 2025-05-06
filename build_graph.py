from Bio.PDB.PDBParser import PDBParser
import numpy as np
import glob
import sys, os
import pickle as pkl
import argparse 
import torch
from torch_geometric.data import Data, Batch
from utils import *
import pandas as pd
# import esm
import gzip


device = "cuda" if torch.cuda.is_available() else "cpu"

target_path = 'data/BindingDB/targets/target.csv'
esmfold_path = 'data/BindingDB/targets/esmfold'
protein_feature_path = 'data/BindingDB/targets/prot_rep.pkl'

with open(protein_feature_path, 'rb') as f:
    protein_feature = pkl.load(f)


# targetfeature_dict = {}

target_df = pd.read_csv(target_path)

target_ids = target_df['Target_ID']

targetesm_dict = dict(zip(target_ids, protein_feature))

for targetid in target_ids:
    target_pdbfile = f'{targetid}.pdb'
    esmfoldfile_path = os.path.join(esmfold_path, target_pdbfile)

        
    if os.path.exists(esmfoldfile_path):
        
        dis_map, seq = load_predicted_PDB3(esmfoldfile_path)
        
        emb = targetesm_dict[targetid]
        emb = emb[0]
        emb = emb[1 : len(emb) - 1]
        if len(seq) == len(emb):
                    
            row, col = np.where(dis_map <= 8)
            edge = [row, col]
            graph = protein_graph(seq, edge, emb)
            
            torch.save(graph, f'data/BACE1/graph/{targetid}.pt')
            
            # targetfeature_dict[targetid] = graph
        else:
            print(targetid)
            print('error')
            print(len(seq))
            print(len(emb))
        
    else:
        continue
    
