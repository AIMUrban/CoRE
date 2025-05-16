import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import math
import random
import sys


class CoRE(nn.Module):
    def __init__(self, s_region_num, t_region_num, s_graph_info, t_graph_info,
                 hidden_dim, gat_layers=2, num_heads=8):
        super().__init__()

        self.s_region_emb = nn.Embedding(s_region_num, hidden_dim)
        self.t_region_emb = nn.Embedding(t_region_num, hidden_dim)
        self.s_edge_index, self.s_edge_weight, self.s_mob = s_graph_info['edge_index'], s_graph_info['edge_weight'], \
                                                            s_graph_info['mob']
        self.t_edge_index, self.t_edge_weight, self.t_mob = t_graph_info['edge_index'], t_graph_info['edge_weight'], \
                                                            t_graph_info['mob']

        self.gat_layers = gat_layers
        self.gat_s = nn.ModuleList([GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=8, concat=False, edge_dim=1)
                                          for _ in range(self.gat_layers)])
        self.gat_t = nn.ModuleList([GATConv(in_channels=hidden_dim, out_channels=hidden_dim, heads=8, concat=False, edge_dim=1)
                                          for _ in range(self.gat_layers)])

        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)

        self.transfer_linear = nn.Linear(hidden_dim, hidden_dim)

        self.s_linear = nn.Linear(hidden_dim, hidden_dim)
        self.activation = torch.nn.PReLU()

        self.cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-8)

    def get_region_emb(self):
        s_region_emb = self.s_region_emb.weight
        t_region_emb = self.t_region_emb.weight

        for i in range(self.gat_layers - 1):
            s_region_emb = self.gat_s[i](s_region_emb, self.s_edge_index, self.s_edge_weight)
            s_region_emb = self.activation(s_region_emb)
            t_region_emb = self.gat_t[i](t_region_emb, self.t_edge_index, self.t_edge_weight)
            t_region_emb = self.activation(t_region_emb)
        s_region_emb = self.gat_s[self.gat_layers - 1](s_region_emb, self.s_edge_index, self.s_edge_weight)
        t_region_emb = self.gat_t[self.gat_layers - 1](t_region_emb, self.t_edge_index, self.t_edge_weight)

        return s_region_emb, t_region_emb

    def forward(self, sample_num):
        s_region_emb, t_region_emb = self.get_region_emb()
        s_intraCity_loss = self.mobility_prediction_loss(s_region_emb, s_region_emb, self.s_mob)
        t_intraCity_loss = self.mobility_prediction_loss(t_region_emb, t_region_emb, self.t_mob)

        crossAlign_loss = self.cross_align(s_region_emb, t_region_emb, sample_num=sample_num)

        crossRec_loss = self.cross_rec(s_region_emb, t_region_emb,)

        return s_intraCity_loss, t_intraCity_loss, crossRec_loss, crossAlign_loss

    def cross_align(self, s_emb, t_emb, sample_num=1000):
        s_emb_sampled = s_emb
        t_emb_sampled = t_emb
        query_emb = torch.randn(sample_num, s_emb.shape[1])

        sim_s = torch.matmul(query_emb, s_emb_sampled.T)
        sim_t = torch.matmul(query_emb, t_emb_sampled.T)
        sim_s = F.normalize(sim_s, dim=-1)
        sim_t = F.normalize(sim_t, dim=-1)

        dif_s = 1 - torch.matmul(sim_s, sim_s.T)
        dif_t = 1 - torch.matmul(sim_t, sim_t.T)

        sem_loss = torch.mean(F.pairwise_distance(dif_s, dif_t, p=2) ** 2)
        return sem_loss

    def cross_rec(self, s_region_emb, t_region_emb):
        s_q, s_k = self.wq(s_region_emb), self.wk(s_region_emb)
        t_q, t_k = self.wq(t_region_emb), self.wk(t_region_emb)

        # s_attn_weight = F.normalize(torch.matmul(s_q, s_k.T) / math.sqrt(s_k.shape[1]), dim=-1)
        # t_attn_weight = F.normalize(torch.matmul(t_q, t_k.T) / math.sqrt(t_k.shape[1]), dim=-1)
        # st_attn_weight = F.normalize(torch.matmul(s_q, t_k.T) / math.sqrt(t_k.shape[1]), dim=-1)
        # ts_attn_weight = F.normalize(torch.matmul(t_q, s_k.T) / math.sqrt(s_k.shape[1]), dim=-1)
        # s_circle_score = F.normalize(torch.matmul(st_attn_weight, ts_attn_weight), dim=-1)
        # t_circle_score = F.normalize(torch.matmul(ts_attn_weight, st_attn_weight), dim=-1)
        s_attn_weight = torch.matmul(s_q, s_k.T) / math.sqrt(s_k.shape[1])
        t_attn_weight = torch.matmul(t_q, t_k.T) / math.sqrt(t_k.shape[1])
        st_attn_weight = torch.matmul(s_q, t_k.T) / math.sqrt(t_k.shape[1])
        ts_attn_weight = torch.matmul(t_q, s_k.T) / math.sqrt(s_k.shape[1])
        s_circle_score = torch.matmul(st_attn_weight, ts_attn_weight)
        t_circle_score = torch.matmul(ts_attn_weight, st_attn_weight)

        s_recon_loss = torch.mean(F.pairwise_distance(s_circle_score, s_attn_weight, p=2) ** 2)
        t_recon_loss = torch.mean(F.pairwise_distance(t_circle_score, t_attn_weight, p=2) ** 2)

        cross_rec_loss = s_recon_loss + t_recon_loss
        return cross_rec_loss

    def mobility_prediction_loss(self, s_embeds: torch.Tensor, d_embeds: torch.Tensor, mob):
        mask = torch.tensor(mob != 0, dtype=torch.bool)
        mob = torch.tensor(mob)

        inner_prod_s = torch.matmul(s_embeds, d_embeds.transpose(-2, -1))
        ps_hat = F.log_softmax(inner_prod_s, dim=-1)
        ps_hat = torch.masked_select(ps_hat, mask)
        mob_s = torch.masked_select(mob, mask)

        loss = torch.sum(-torch.mul(mob_s, ps_hat))
        return loss
