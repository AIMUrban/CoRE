import numpy as np
import tqdm
from Model.CoRE import *
from evaluator.evaluator_cross_city import Evaluator
from torch.utils.data import DataLoader
import pickle
import torch
from utils import *
from optim_schedule import ScheduledOptim
import pandas as pd
import random
# import faiss
from data import *
import argparse
import os


if __name__ == '__main__':
    # set some parameters
    epochs = 600
    d_feature = 128

    exp_id = int(random.SystemRandom().random() * 1000000)
    exp_cache_file = './output/{}'.format(exp_id)
    ensure_cache_dir(exp_cache_file)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load and preprocess datas
    s_dataset = 'bj'
    t_dataset = 'xa'
    print('learning {} --- {}'.format(s_dataset, t_dataset))
    s_region_num, s_graph_info = load_data(s_dataset)
    t_region_num, t_graph_info = load_data(t_dataset)

    model = CoRE(s_region_num=s_region_num, t_region_num=t_region_num, s_graph_info=s_graph_info,
                 t_graph_info=t_graph_info, hidden_dim=d_feature, gat_layers=2, num_heads=8)

    evaluator = Evaluator(exp_id=exp_id)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    results = []
    min_loss = float('inf')
    best_epoch = 0
    best_s_emb, best_t_emb = None, None
    save_path = exp_cache_file + '/train_cache'

    for epoch_id in range(epochs):
        model.train()

        s_intraCity_loss, t_intraCity_loss, crossRec_loss, crossAlign_loss = model.forward(sample_num=1000)

        all_loss = (s_intraCity_loss + t_intraCity_loss) + crossRec_loss + crossAlign_loss

        print('Epoch {}/{}, all_loss={:.5f}'.format(
                epoch_id + 1, epochs, all_loss.item(), ))

        optim.zero_grad()
        all_loss.backward()
        optim.step()

        if (epoch_id + 1) % 50 == 0 or epoch_id == 0:
            best_epoch = epoch_id + 1

            s_emb, t_emb = model.get_region_emb()
            s_emb, t_emb = s_emb.detach().cpu().numpy(), t_emb.detach().cpu().numpy()
            print(save_path)
            np.save(save_path + '/{}_region_emb_{}.npy'.format(
                s_dataset, best_epoch), s_emb)
            np.save(save_path + '/{}_region_emb_{}.npy'.format(
                t_dataset, best_epoch), t_emb)

            evaluator.evaluate(s_dataset=s_dataset, t_dataset=t_dataset, s_region_emb=s_emb,
                               t_region_emb=t_emb)
