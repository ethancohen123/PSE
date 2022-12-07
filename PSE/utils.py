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


def graph_info(graph):
    G = graph
    print('We have %d nodes.' % G.number_of_nodes())
    print('We have %d edges.' % G.number_of_edges())
    print('Drawing the graph network')
    nx_G = G.to_networkx().to_undirected()
    pos = nx.kamada_kawai_layout(nx_G)
    options = { "node_color": "purple",
                "edge_color": "gray",
                "node_size": 10,
                "linewidths": 0,
                "width": 0.1 }
    nx.draw(nx_G, pos, **options)


#Perdicts the DTI using pretrained models
def calc_affinity(drug, target, pre_model='MPNN_CNN_DAVIS'):
    '''
    Perdicts the Drug-Target Interaction (DTI)
    affinity using pretrained DeepPurpose models.

        1. SMILES string of the drug.
        2. Amino Acid Sequence of the target.
        3. Default model = 'MPNN_CNN_DAVIS'

    '''
    model_info = pre_model.split('_') # Seperates data and encodings
    model = models.model_pretrained(model=pre_model)
    X_pred = utils.data_process(X_drug=[drug], X_target=[target], y=[0],
                                drug_encoding=model_info[0], target_encoding=model_info[1],
                                split_method='no_split')
    y_pred = model.predict(X_pred)  # Perdicts affinity
    return str(round(y_pred[0], 2))


def smi2psa():
    return()

def pred2prot():
    return()
