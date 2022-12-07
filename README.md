# Polypharmacy Side Effect (PSE) prediction

As part of [master's thesis](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-446691), I have developed a chemistry-informed and interpretable Siamaes Graph Convolutional Network (SiGCN) model that predicts polypharmacy side effects of drug combinations based on their SMILES. The following repository is a pollished vesion of the original [source code](https://github.com/amanzadi/PSE/src) that you can easily test and predict PSEs with using the pretrained models.

![Architecture of the Siamese Graph Convolutional Neural Network (SiGN)](/figs/SiGCN.png)


## Installation

### Build from Source

```bash
git clone https://github.com/amanzadi/PSE.git ## Download code repository
cd PSE ## Change directory to PSE
pip install -r requirements.txt
```

## Example

### 1. Predict PSE of FDA drugs with pretrained models (fast)

The `fda_drugs` module includes 1430 FDA approved drug on the and their corresponding Protein-Side effect Association (PSA) graphs can be accessed with `psa_graph` module. You can predict any number of combinations but the model have been primaraly evaluated only for pairwised drug combinations. Since the PSA graphs have already constructed, and the PSE and be predicted within seconds. 


``` python
from PSE import fda_drug, psa_subgraph, pse_predict

# return the list of FDA approved drugs
fda_drug_list = fda_drug(all)

# extracts the coresponding Protein-Side effect association (PSA) graphs
g1 = psa_subgraph(fda_drug['drug_name_1'])
g2 = psa_subgraph(fda_drug['drug_name_2'])
  
# list of drugs in the combination
combs = [g1,g2] 
  
# loads amd executes the pre-trained SiGCN model  
pse_predict(combs)
```


### 2. Predict PSE from SMILES with pretrained models (slow)

If you want to find PSE of non-FDA approved drug, the PSA graphs have to be created from scratch with `smi2psa` . That is why this method will take much longer to predict PSE.


``` python
from PSE import smi2psa_subgraph, pse_predict
  
#genrates the PSA graph from SMILES string 
g1 = smi2psa_subgraph('SMILES_1')
g2 = smi2psa_subgraph('SMILES_2')

# list of drugs in the combination
combs = [g1,g2] 
  
# loads amd executes the pre-trained SiGCN model  
pse_predict(combs)
```


### 3. Interpret PSE perdiction

Extracts and visulizes the common protein in PSA that could be responsible for the predicted PSEs using `pred2prot`.
  
``` python
from PSE import fda_drug, psa_graph, ppse_predict, pred2prot

# return the FDA approved drug dictionary
fda_drug_list = fda_drug(all)

# extracts the corespondin Protein-Side effect association (PSA) graphs
g1,g2 = psa_subgraph(fda_drug['drug_name_1']), psa_subgraph(fda_drug['drug_name_2'])


# list of drugs in the combination
combs = [g1,g2] 
  
# loads amd executes the pre-trained SiGCN model  
PSEs = pse_predict(combs)  
  
# return the common PSA network
PSAs = pred2prot(combs)
```

## Bibliography

If you found this package useful, please cite [my thesis](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-446691):
```
@article{amanzadi2021pse,
  title={Predicting safe drug combinations with Graph Neural Networks (GNN)},
  author={Amirhossein Amanzadi},
  journal={Uppsala University},
  year={2021}
}
```

## Disclaimer
The output list should be inspected manually by experts before proceeding to the wet-lab validation, and our work is still in active developement with limitations, please do not directly use the drugs.
