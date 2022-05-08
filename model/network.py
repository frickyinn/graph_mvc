#! /usr/bin/env python
# -*- coding:utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F

from .layers import GraphConvolution, InnerProductDecoder, full_block


class Encoder(nn.Module):
    def __init__(self, input_dim, linear_dims=[1024, 128, 32], gcn_dims=[32, 8], p_drop=0.2):
        super(Encoder, self).__init__()
        self.linear = nn.Sequential(
            # nn.Dropout(p_drop),
            # nn.Linear(input_dim, linear_dims[0]),
            # nn.ReLU(),
            #
            # nn.Dropout(p_drop),
            # nn.Linear(linear_dims[0], linear_dims[1]),
            # nn.ReLU(),
            #
            # nn.Dropout(p_drop),
            # nn.Linear(linear_dims[1], linear_dims[2]),
            # nn.ReLU(),

            full_block(input_dim, linear_dims[1], p_drop),
            full_block(linear_dims[1], linear_dims[2], p_drop)
        )
        self.gc1 = GraphConvolution(linear_dims[2], gcn_dims[0], p_drop, act=F.relu)
        self.gc2 = GraphConvolution(gcn_dims[0], gcn_dims[1], p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(gcn_dims[0], gcn_dims[1], p_drop, act=lambda x: x)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        x = self.linear(x)
        g = self.gc1(x, adj)
        mu, logvar = self.gc2(g, adj), self.gc3(g, adj)
        g = self.reparameterize(mu, logvar)
        x = torch.cat((x, g), dim=1)

        return x, g, mu, logvar


class Decoder(nn.Module):
    def __init__(self, input_dim, linear_dims=[1024, 128, 32], gcn_dims=[32, 8], p_drop=0.2):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(
            # nn.Dropout(p_drop),
            # nn.Linear(linear_dims[2]+gcn_dims[1], linear_dims[1]),
            # nn.ReLU(),
            #
            # nn.Dropout(p_drop),
            # nn.Linear(linear_dims[1], input_dim),
            full_block(linear_dims[2]+gcn_dims[1], input_dim, p_drop)
        )
        self.dc = InnerProductDecoder(p_drop, act=lambda x: x)

    def forward(self, x):
        xrl = self.linear(x)
        xrg = self.dc(x)

        return xrl, xrg


class Network(nn.Module):
    def __init__(self, view, input_dims, class_num, linear_dims=[1024, 128, 32], gcn_dims=[32, 8],
                 high_feature_dim=32, p_drop=0.2, device='cpu'):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_dims[v], linear_dims, gcn_dims, p_drop).to(device))
            self.decoders.append(Decoder(input_dims[v], linear_dims, gcn_dims, p_drop).to(device))

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.feature_contrastive_module = nn.Sequential(
            # nn.Linear(linear_dims[2]+gcn_dims[1], high_feature_dim),

            full_block(linear_dims[2]+gcn_dims[1], linear_dims[1], p_drop),
            nn.Linear(linear_dims[1], high_feature_dim),
        )
        self.label_contrastive_module = nn.Sequential(
            # nn.Linear(linear_dims[2]+gcn_dims[1], class_num),

            full_block(linear_dims[2] + gcn_dims[1], linear_dims[1], p_drop),
            nn.Linear(linear_dims[1], class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs, adj):
        hs = []
        qs = []
        xrls = []
        xrgs = []
        zs = []
        mus = []
        logvars = []
        for v in range(self.view):
            x = xs[v]

            z, _, mu, logvar = self.encoders[v](x, adj)
            h = F.normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xrl, xrg = self.decoders[v](z)

            hs.append(h)
            zs.append(z)
            qs.append(q)
            xrls.append(xrl)
            xrgs.append(xrg)
            mus.append(mu)
            logvars.append(logvar)

        return hs, qs, xrls, xrgs, zs, mus, logvars

    # def forward_plot(self, xs):
    #     zs = []
    #     hs = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         z = self.encoders[v](x)
    #         zs.append(z)
    #         h = self.feature_contrastive_module(z)
    #         hs.append(h)
    #     return zs, hs
    #
    # def forward_cluster(self, xs):
    #     qs = []
    #     preds = []
    #     for v in range(self.view):
    #         x = xs[v]
    #         z = self.encoders[v](x)
    #         q = self.label_contrastive_module(z)
    #         pred = torch.argmax(q, dim=1)
    #         qs.append(q)
    #         preds.append(pred)
    #     return qs, preds

