# Polypharmacy Side Effect (PSE) prediction

## Installing PSE

### 1. Conda 

```bash
git clone https://github.com/amanzadi/PSE.git ## Download code repository
cd PSE ## Change directory to PSE
conda env create -f environment.yml  ## Build virtual environment with all packages installed using conda
conda activate PSE ## Activate conda environment (use "source activate PSE" for anaconda 4.4 or earlier) 
jupyter notebook ## open the jupyter notebook with the conda env
```

### 2. pip

```bash
git clone https://github.com/amanzadi/PSE.git ## Download code repository
cd PSE ## Change directory to PSE
pip install -r requirements.txt
```

## Predict PSE of FDA drugs unising pretrained models

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
