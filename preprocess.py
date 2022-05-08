#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np

from utils.st import load_ST_file, adata_preprocess
from utils.graph import graph_construction


DATA_PATH = '../spatialLIBD'
NAME = '151673'

path = os.path.join(DATA_PATH, NAME)
meta_path = os.path.join(path, 'metadata.tsv')
n_top_genes = 4000
k = 20

adata_h5 = load_ST_file(path)
adata_X = adata_preprocess(adata_h5, min_cells=5, n_top_genes=n_top_genes)
graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], k=k)

if not os.path.exists(f'./data/{NAME}'):
    os.mkdir(f'./data/{NAME}')

np.save(f'./data/{NAME}/adatax_hvg{n_top_genes}.npy', adata_X)
np.save(f'./data/{NAME}/graphdict_hvg{n_top_genes}.npy', graph_dict, allow_pickle=True)
os.system(f'cp {meta_path} ./data/{NAME}/metadata.tsv')
