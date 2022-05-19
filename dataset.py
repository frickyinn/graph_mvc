#! /usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
import torch
from torch.utils import data


class SpatialLIBD(data.Dataset):
    def __init__(self, path):
        self.x1 = np.load(f'{path}/adatax_pca300.npy').astype(np.float32)
#         self.x2 = np.load(f'{path}/extracted_feature.npy').astype(np.float32)
#         self.x2 = np.load(f'{path}/151673/151673_mae_imagenet_1024.npy').astype(np.float32)
        self.x2 = np.load(f'{path}/151673/151673_mae_imagenet_spatialLIBD_1024.npy').astype(np.float32)
        
        graph_dict = np.load(f'{path}/graphdict_pca300.npy', allow_pickle=True).item()
        df_meta = pd.read_csv(f'{path}/metadata.tsv', sep='\t')

        # mask = ~pd.isnull(df_meta['layer_guess']).values
        # self.adj_norm = graph_dict["adj_norm"][mask].T[mask].T
        # self.adj_label = graph_dict["adj_label"][mask].T[mask].T
        # self.norm_value = graph_dict["norm_value"]
        #
        # self.gene_expr = self.gene_expr[mask]
        # self.mae_feat = self.mae_feat[mask]
        # df_meta = df_meta[mask]

        self.adj_norm = graph_dict["adj_norm"]
        self.adj_label = graph_dict["adj_label"]
        self.norm_value = graph_dict["norm_value"]

        self.labels = pd.Categorical(df_meta['layer_guess']).codes
        self.len = len(self.labels)

        self.xs = [self.x1, self.x1, self.x2, self.x2]
        self.dims = [x.shape[1] for x in self.xs]
        self.view = 4
        self.class_num = 7

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        x1 = self.gene_expr[idx]
        x2 = self.mae_feat[idx]

        return [torch.from_numpy(x1), torch.from_numpy(x2)], torch.tensor(self.labels[idx]).long(), torch.from_numpy(
            np.array(idx)).long()


# def load_spatial_data(name):
#     dataset = SpatialLIBD(f'./data/spatialLIBD/{name}')
#     dims = [300, 768]
#     view = 2
#     data_size = 3611
#     class_num = 7
#
#     return dataset, dims, view, data_size, class_num
