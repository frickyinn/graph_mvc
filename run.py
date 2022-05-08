#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import torch
import random
import numpy as np

from dataset import SpatialLIBD
from model.network import Network
from trainer import Trainer


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch(1)
lr = 0.001

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = 'cuda'

name = '151673'
save_path = './checkpoints'
log_dir = './log/init'

# print(os.getcwd())
dataset = SpatialLIBD(f'/home/yinrui/graph_mvc/data/{name}')
network = Network(dataset.view, dataset.dims, dataset.class_num, linear_dims=[1024, 128, 32], gcn_dims=[32, 8],
                  high_feature_dim=32, p_drop=0.2, device=device)
optimizer = torch.optim.Adam(list(network.parameters()), lr=lr)
model = Trainer(network, optimizer, save_path=save_path, log_dir=log_dir, device=device)

model.fit(dataset)
