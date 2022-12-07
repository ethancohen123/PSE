# ============================ 1. Environment setup ============================
import dgl
import torch
import pickle
import random as rd
import gradio as gr
import numpy as np
import pandas as pd
import networkx as nx
import torch.nn as nn
from tqdm.notebook import tqdm
import torch.nn.functional as F
from dgl.data import DGLDataset
from DeepPurpose import utils
from DeepPurpose import DTI as models


# ================================ 2. Functions =================================


# model
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats)
        self.conv2 = GraphConv(h_feats,  num_classes)
        
    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        g.ndata['h'] = h
        out = F.relu(dgl.mean_nodes(g, 'h'))
        #out = F.relu(dgl.max_nodes(g, 'h'))
        return out


def pse_predict(Drug_A, Drug_B):
    
    g1 = drug_graphs[drug_labels.index(Drug_A)]
    g2 = drug_graphs[drug_labels.index(Drug_B)]
    pred1 = model(g1, g1.ndata['PSE'].float())
    pred2 = model(g2, g2.ndata['PSE'].float())
    pred_sum = pred1+pred2
    pred = ((pred_sum)*0.5)/pred_sum.max()
    pred_PSE = pred.ne(0)[0].tolist()  # not equal to 0
    pred_PSE_value = pred[0].tolist()

    tmp = []
    for idx, se in enumerate(pred_PSE):
        if se == True:
            tmp.append([PSE_dic[idx], round(pred_PSE_value[idx], 3)])

    df = pd.DataFrame(tmp, columns=['PSE', 'Value'])
    #df['Value'] = df['Value']/df['Value'].max()
    df = df.sort_values(by=['Value'], ascending=False)
    df = df.iloc[15:]
    df = df.reset_index(drop=True)
    dic = {df['PSE'][i]: df['Value'][i] for i in rd.sample(range(len(df)), 15)}
    #dic = {df['PSE'][i]:df['Value'][i] for i in range(len(df))}
    #print(df)
    return(dic)    
# Specify a path
conf = '../data/sigcn.pt'

# Load
model = GCN(964,200,964)
model.load_state_dict(torch.load(conf))
model.eval()