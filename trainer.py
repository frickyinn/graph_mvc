#! /usr/bin/env python
# -*- coding:utf-8 -*-
import os
import anndata
import scanpy as sc
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
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


def eval_kmeans_ari(labels, z):
    kmeans = KMeans(7)
    kmeans.fit(z)
    preds = kmeans.predict(z)
    
    preds = preds[labels != -1]
    labels = labels[labels != -1]
    
    return adjusted_rand_score(labels, preds)


def eval_pca_kmeans_ari(labels, z):
    pca = PCA(100)
    z = pca.fit_transform(z)
#     print(z.shape)
    kmeans = KMeans(7)
    kmeans.fit(z)
    preds = kmeans.predict(z)
    
    preds = preds[labels != -1]
    labels = labels[labels != -1]
    
    return adjusted_rand_score(labels, preds)


class Trainer:
    def __init__(self, network, optimizer, save_path, log_dir, device='cpu', vis_view=-1):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir)
        # self.pretrain_writer = SummaryWriter(os.path.join(log_dir, 'pretrain'))
        # self.contrastive_writer = SummaryWriter(os.path.join(log_dir, 'contrastive'))
        # self.fine_tune_writer = SummaryWriter(os.path.join(log_dir, 'fine_tune'))

        self.network = network.to(device)
        self.device = device
        self.optimizer = optimizer

        self.vis_view = vis_view

    def pretrain(self, dataset, epoch):
        self.network.train()
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

#         ari = eval_kmeans_ari(dataset.labels, dataset.xs[0].detach().cpu().numpy())
#         ari = eval_kmeans_ari(dataset.labels, zs[0].detach().cpu().numpy())
        if self.vis_view == -1:
            ari = adjusted_rand_score(dataset.labels, ((qs[1] + qs[0]) / 2).argmax(dim=1).detach().cpu().numpy())
        else:
            ari = adjusted_rand_score(dataset.labels, qs[self.vis_view].argmax(dim=1).detach().cpu().numpy())

        print('Pretrain Epoch {}'.format(epoch), 'loss:{:.4f}'.format(loss.item()), 'q_ari:{:.4f}'.format(ari))
        self.writer.add_scalar('loss', loss.item(), epoch)
        self.writer.add_scalar('ari', ari, epoch)
        self.writer.flush()

    def contrastive_train(self, dataset, epoch):
        self.network.train()
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

#         ari = eval_kmeans_ari(dataset.labels, dataset.xs[0].detach().cpu().numpy())
#         ari = eval_kmeans_ari(dataset.labels, zs[0].detach().cpu().numpy())
        if self.vis_view == -1:
            ari = adjusted_rand_score(dataset.labels, ((qs[1] + qs[0]) / 2).argmax(dim=1).detach().cpu().numpy())
        else:
            ari = adjusted_rand_score(dataset.labels, qs[self.vis_view].argmax(dim=1).detach().cpu().numpy())

        print('Contrastive Epoch {}'.format(epoch), 'loss:{:.4f}'.format(loss.item()), 'q_ari:{:.4f}'.format(ari))
        self.writer.add_scalar('loss', loss.item(), epoch)
        # self.contrastive_writer.add_scalar('h_ari', h_ari, epoch)
        self.writer.add_scalar('ari', ari, epoch)
        self.writer.flush()

    def make_pseudo_label(self, dataset):
        self.network.eval()
        scaler = MinMaxScaler()
        with torch.no_grad():
            hs, _, _, _, _, _, _ = self.network(dataset.xs, dataset.adj_norm)
        for v in range(dataset.view):
            hs[v] = hs[v].detach().cpu().numpy()
            hs[v] = scaler.fit_transform(hs[v])

        kmeans = KMeans(n_clusters=dataset.class_num, n_init=100)
        new_pseudo_label = []
        for v in range(dataset.view):
            pseudo_label = kmeans.fit_predict(hs[v])
            pseudo_label = pseudo_label.reshape(len(dataset), 1)
            pseudo_label = torch.from_numpy(pseudo_label)
            new_pseudo_label.append(pseudo_label)

        return new_pseudo_label

    def match(self, y_true, y_pred):
        y_true = y_true.astype(np.int64)
        y_pred = y_pred.astype(np.int64)
        assert y_pred.size == y_true.size
        D = max(y_pred.max(), y_true.max()) + 1
        w = np.zeros((D, D), dtype=np.int64)
        for i in range(y_pred.size):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)
        new_y = np.zeros(y_true.shape[0])
        for i in range(y_pred.size):
            for j in row_ind:
                if y_true[i] == col_ind[j]:
                    new_y[i] = row_ind[j]
        new_y = torch.from_numpy(new_y).long().to(self.device)
        new_y = new_y.view(new_y.size()[0])
        return new_y

    def fine_tuning(self, dataset, new_pseudo_label, epoch):
        self.network.train()
        cross_entropy_criterion = torch.nn.CrossEntropyLoss()

        self.optimizer.zero_grad()
        hs, qs, xrls, xrgs, zs, mus, logvars = self.network(dataset.xs, dataset.adj_norm)
        loss_list = []
        for v in range(dataset.view):
            p = new_pseudo_label[v].numpy().T[0]
            with torch.no_grad():
                q = qs[v].detach().cpu()
                q = torch.argmax(q, dim=1).numpy()
                p_hat = self.match(p, q)
            loss_list.append(cross_entropy_criterion(qs[v], p_hat))

        loss = sum(loss_list)
        loss.backward()
        self.optimizer.step()

#         ari = eval_kmeans_ari(dataset.labels, dataset.xs[0].detach().cpu().numpy())
#         ari = eval_kmeans_ari(dataset.labels, zs[0].detach().cpu().numpy())
        if self.vis_view == -1:
            ari = adjusted_rand_score(dataset.labels, ((qs[1] + qs[0]) / 2).argmax(dim=1).detach().cpu().numpy())
        else:
            ari = adjusted_rand_score(dataset.labels, qs[self.vis_view].argmax(dim=1).detach().cpu().numpy())

        print('Contrastive Epoch {}'.format(epoch), 'loss:{:.4f}'.format(loss.item()), 'q_ari:{:.4f}'.format(ari))
        self.writer.add_scalar('loss', loss.item(), epoch)
        self.writer.add_scalar('ari', ari, epoch)
        self.writer.flush()

    def fit(self, dataset, pretrain_epochs=200, contrastive_epochs=300, fine_tune_epochs=100):
        dataset.adj_norm = dataset.adj_norm.to(self.device)
        dataset.adj_label = dataset.adj_label.to(self.device)

        for v in range(dataset.view):
            dataset.xs[v] = Variable(torch.from_numpy(dataset.xs[v])).to(self.device)
        # dataset.labels = torch.from_numpy(dataset.labels).long()

        for epoch in range(pretrain_epochs):
            self.pretrain(dataset, epoch)

        for epoch in range(contrastive_epochs):
            self.contrastive_train(dataset, epoch + pretrain_epochs)

        new_pseudo_label = self.make_pseudo_label(dataset)
        for epoch in range(fine_tune_epochs):
            self.fine_tuning(dataset, new_pseudo_label, epoch+pretrain_epochs+contrastive_epochs)
