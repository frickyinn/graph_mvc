#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import anndata
import scanpy as sc
from sklearn.metrics import adjusted_rand_score
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from loss import gcn_criterion, ContrastiveLoss


def eval_leiden_ari(labels, z):
    adata = anndata.AnnData(z)
    sc.pp.neighbors(adata, n_neighbors=20)
    cluster_key = "leiden"
    sc.tl.leiden(adata, key_added=cluster_key, resolution=1)

    preds = adata.obs['leiden'].values
    preds = preds[labels != -1]
    labels = labels[labels != -1]

    return adjusted_rand_score(labels, preds)


class Trainer:
    def __init__(self, network, optimizer, save_path, log_dir, device='cpu'):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path
        self.pretrain_writer = SummaryWriter(os.path.join(log_dir, 'pretrain'))
        self.contrastive_writer = SummaryWriter(os.path.join(log_dir, 'contrastive'))
        self.fine_tune_writer = SummaryWriter(os.path.join(log_dir, 'fine_tune'))

        self.network = network.to(device)
        self.device = device
        self.optimizer = optimizer

    def pretrain(self, dataset, epoch):
        rec_criterion = nn.MSELoss()

        self.optimizer.zero_grad()
        hs, qs, xrls, xrgs, zs, mus, logvars = self.network(dataset.xs, dataset.adj_norm)
        loss_list = []
        for v in range(dataset.view):
            loss_gcn = gcn_criterion(preds=xrgs[v], labels=dataset.adj_label, mu=mus[v], logvar=logvars[v],
                                     n_nodes=len(dataset), norm=dataset.norm_value, mask=dataset.adj_label)
            loss_rec = rec_criterion(dataset.xs[v], xrls[v])
            loss_list.append(10 * loss_rec + 0.1 * loss_gcn)

        loss = sum(loss_list)
        loss.backward()
        self.optimizer.step()

        ari = eval_leiden_ari(dataset.labels, zs[0].detach().cpu().numpy())

        print('Pretrain Epoch {}'.format(epoch), 'loss:{:.4f}'.format(loss.item()), 'ari:{:.4f}'.format(ari))
        self.pretrain_writer.add_scalar('loss', loss.item(), epoch)
        self.pretrain_writer.add_scalar('ari', ari, epoch)
        self.pretrain_writer.flush()

    def contrastive_train(self, dataset, epoch):
        rec_criterion = torch.nn.MSELoss()
        con_criterion = ContrastiveLoss(len(dataset), dataset.class_num, 0.5, 1.0, self.device).to(self.device)

        self.optimizer.zero_grad()
        hs, qs, xrls, xrgs, zs, mus, logvars = self.network(dataset.xs, dataset.adj_norm)
        loss_list = []
        for v in range(dataset.view):
            loss_gcn = gcn_criterion(preds=xrgs[v], labels=dataset.adj_label, mu=mus[v], logvar=logvars[v],
                                     n_nodes=len(dataset), norm=dataset.norm_value, mask=dataset.adj_label)
            loss_rec = rec_criterion(dataset.xs[v], xrls[v])
            loss_list.append((10 * loss_rec + 0.1 * loss_gcn) * 0.1)
            for w in range(v+1, dataset.view):
                loss_list.append(con_criterion.forward_feature(hs[v], hs[w]))
                loss_list.append(con_criterion.forward_label(qs[v], qs[w]))

        loss = sum(loss_list)
        loss.backward()
        self.optimizer.step()

        # h_ari = eval_leiden_ari(dataset.labels, hs[0].detach().cpu().numpy())
        q_ari = adjusted_rand_score(dataset.labels, ((qs[1]+qs[0])/2).argmax(dim=1).detach().cpu().numpy())

        print('Contrastive Epoch {}'.format(epoch), 'loss:{:.4f}'.format(loss.item()), 'q_ari:{:.4f}'.format(q_ari))
        self.contrastive_writer.add_scalar('loss', loss.item(), epoch)
        # self.contrastive_writer.add_scalar('h_ari', h_ari, epoch)
        self.contrastive_writer.add_scalar('ari', q_ari, epoch)
        self.contrastive_writer.flush()

    def fit(self, dataset, pretrain_epochs=200, contrastive_epochs=300):
        dataset.adj_norm = dataset.adj_norm.to(self.device)
        dataset.adj_label = dataset.adj_label.to(self.device)

        for v in range(dataset.view):
            dataset.xs[v] = Variable(torch.from_numpy(dataset.xs[v])).to(self.device)
        # dataset.labels = torch.from_numpy(dataset.labels).long()

        for epoch in range(pretrain_epochs):
            self.pretrain(dataset, epoch)

        for epoch in range(contrastive_epochs):
            self.contrastive_train(dataset, epoch + pretrain_epochs)

