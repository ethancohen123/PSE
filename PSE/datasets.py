
import os
import wget
import random as rd
import gradio as gr
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

'''
Acknowledgement:
The SIDER datasets are hosted in http://sideeffects.embl.de/download/.
The STRING datasets can be found in https://string-db.org/cgi/download.
The drug SMILES can be found in https://pubchem.ncbi.nlm.nih.gov/docs/downloads/.
We use the side effect categories are hosted in http://snap.stanford.edu/decagon/bio-decagon-effectcategories.tar.gz.
'''

   
# looks for a FDA_drug and returns its PSA's Graph index
def fda_drugs(drug_name):
    '''
    Returns the GraphID of the sprcific fda drug,
    Or returns the entire FDA library.

        1. drug name. Type "all" to get the full list.
    '''
    # loads the fda library
    df_fda_drugs = pd.read_csv('../data/fda_drugs.csv', sep=',')
    
    if drug_name == 'all':
        return(df_fda_drugs['Name'].tolist())
    
    elif drug_name in df_fda_drugs['Name'].tolist():
        return(df_fda_drugs['GraphID'].loc[df_fda_drugs['Name'] == drug_name].value())
    
    elif drug_name not in df_fda_drugs['Name'].tolist():
        print('Can not find this drug name in the FDA library.')
    
    else:
        print('Please enter the correct FDA drug name. Or type "all" to see the entirety of the FDA drug library.')
        
# downloads a given file from the zendoo host
def download_df(path='./data', file_name):    
	print('Beginning Processing...')

	if not os.path.exists(path):
	    os.makedirs(path)

	url = 'https://zendoo...'+file_name
	saved_path = wget.download(url, path)

def check_file_existance(file_name):
    if file_name

# Node file
nodes = pd.read_csv('../data/GNN-GSE_full_pkd_norm.csv', sep=',')

# Edge file
edges = pd.read_csv('../data/GNN-PPI-net.csv', sep=',')

# Drug-protein file (DTI)
dti = pd.read_csv('../data/GNN-DTI_full.csv', sep=',')

# DrugIds
DrugID = pd.read_csv('../data/DrugID.csv', sep=',')


def psa_graph():
    return()