# Polypharmacy Side Effect (PSE) prediction

As part of [master's thesis](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-446691), I have developed a chemistry-informed and interpretable Siamaes Graph Convolutional Network (SiGCN) model that predicts polypharmacy side effects of drug combinations based on their SMILES. The following repository is a pollished vesion of the original [source code](https://github.com/amanzadi/PSE/src) that you can easily test and predict PSEs with using the pretrained models.

![Architecture of the Siamese Graph Convolutional Neural Network (SiGN)](/figs/SiGCN.png)


## Cite Us

If you found this package useful, please cite [our paper](http://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-446691):
```
@article{amanzadi2021pse,
  title={Predicting safe drug combinations with Graph Neural Networks (GNN)},
  author={Amirhossein Amanzadi},
  journal={Uppsala University},
  year={2021}
}
```
## Installation

### Build from Source

```bash
git clone https://github.com/amanzadi/PSE.git ## Download code repository
cd PSE ## Change directory to PSE
pip install -r requirements.txt
```

## Example

### 1. Predict PSE of FDA drugs with pretrained models (fast)

<details>
  <summary>Click here for the code!</summary>

``` python
from PSE import utils, model

# return the FDA approved drug dictionary
fda_drug = utils.fda_drugs_name(all)

# extracts the corespondin Gene-Side effect (GSE) graphs
g1 = utils.gse_graph(fda_drugs['drug1'])
g2 = utils.gse_graph(fda_drugs['drug2'])

# loads the pretraoned SiGCN model
pse_model = model.load_sigcn()
```
</details>  

### 2. Predict PSE from SMILES with pretrained models (slow)

<details>
  <summary>Click here for the code!</summary>

``` python
from PSE import utils, model

# return the FDA approved drug dictionary
fda_drug = utils.fda_drugs_name(all)

# extracts the corespondin Gene-Side effect (GSE) graphs
g1 = utils.gse_graph(fda_drugs['drug1'])
g2 = utils.gse_graph(fda_drugs['drug2'])

# loads the pretraoned SiGCN model
pse_model = model.load_sigcn()
```
</details>

### 3. Interpret PSE perdiction

<details>
  <summary>Click here for the code!</summary>
  
``` python
from PSE import utils, model

# return the FDA approved drug dictionary
fda_drug = utils.fda_drugs_name(all)

# extracts the corespondin Gene-Side effect (GSE) graphs
g1 = utils.gse_graph(fda_drugs['drug1'])
g2 = utils.gse_graph(fda_drugs['drug2'])

# loads the pretraoned SiGCN model
pse_model = model.load_sigcn()
```
</details>

### 4. Visualize the Graph Networks

<details>
  <summary>Click here for the code!</summary>

``` python
from PSE import utils, model

# return the FDA approved drug dictionary
fda_drug = utils.fda_drugs_name(all)

# extracts the corespondin Gene-Side effect (GSE) graphs
g1 = utils.gse_graph(fda_drugs['drug1'])
g2 = utils.gse_graph(fda_drugs['drug2'])

# loads the pretraoned SiGCN model
pse_model = model.load_sigcn()
```
</details>

## Disclaimer
The output list should be inspected manually by experts before proceeding to the wet-lab validation, and our work is still in active developement with limitations, please do not directly use the drugs.
