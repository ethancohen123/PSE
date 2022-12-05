# ============================ 1. Environment setup ============================
import pandas as pd
import numpy as np
from DeepPurpose import utils
from DeepPurpose import DTI as models
import seaborn as sns
import matplotlib.pyplot as plt


# ================================ 2. Functions =================================

look4smi, fda_drugs, psa_graph, psa_predict, smi2psa, pred2prot

def loo2smi(SMILES):
    
# looks for FDA_drugs and 
def fda_drug(drug_name):
   
    df_fda_drugs = pd.read_csv('../data/fda_drugs.csv', sep=',')
    
    if drug_name == 'all':
        #dic_drug_names = {df_fda_drugs['DrugID']:df_fda_drugs['Drug_Names'] for drug in df_fda_drugs}
        return(dic_drug_names['Drug_Names'].tolist())
    
    elif drug_name in dic_drug_names['Drug_Names'].tolist():
        return(dic_drug_names.loc[dic_drug_names['Drug_Names'] == drug_name].value())
    
    elif drug_name not in dic_drug_names['Drug_Names'].tolist():
        print('Can not find the drug name in our FDA library.\n Please use "look4smi" to search the FDA drug library with th SMILES string.')
    
    else:
        print('Please enter the correct drug name. Or type "all" to see the entirety of the FDA drug list.')
        
  

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

