#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import numpy as np
import cv2

from utils.st import load_ST_file, adata_preprocess
from utils.graph import graph_construction


DATA_PATH = '../spatialLIBD'
NAME = '151673'

path = os.path.join(DATA_PATH, NAME)
meta_path = os.path.join(path, 'metadata.tsv')
n_top_genes = 5000
k = 20

adata_h5 = load_ST_file(path)
adata_X = adata_preprocess(adata_h5, min_cells=5, n_top_genes=n_top_genes)
graph_dict = graph_construction(adata_h5.obsm['spatial'], adata_h5.shape[0], k=k)

if not os.path.exists(f'./data/{NAME}'):
    os.mkdir(f'./data/{NAME}')

np.save(f'./data/{NAME}/adatax_hvg{n_top_genes}.npy', adata_X)
np.save(f'./data/{NAME}/graphdict_hvg{n_top_genes}.npy', graph_dict, allow_pickle=True)
os.system(f'cp {meta_path} ./data/{NAME}/metadata.tsv')

full_image = cv2.imread(os.path.join(path, f'{NAME}_full_image.tif'))
patches = []
for x, y in adata_h5.obsm['spatial']:
    patches.append(full_image[y-112:y+112, x-112:x+112])
patches = np.array(patches)

PATCHES_PATH = f'./data/{NAME}/patches.npy'
OUTPUT_DIR = f'./data/{NAME}'
np.save(PATCHES_PATH, patches)
os.system(f'python MAE-pytorch/run_mae_extract_feature.py {PATCHES_PATH} {OUTPUT_DIR} {MODEL_PATH}')