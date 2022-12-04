# ============================ 1. Environment setup ============================
import pandas as pd
import numpy as np
from DeepPurpose import utils
from DeepPurpose import DTI as models
import seaborn as sns
import matplotlib.pyplot as plt


# ================================ 2. Functions =================================

fda_drugs, psa_graph

def fda_drug(drug_name):
    if drug_name == 'all':
        return()
    else:
        return()

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

