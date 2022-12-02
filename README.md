# Polypharmacy Side Effect (PSE) prediction

## Installing PSE

```

```


## Predict PSE of FDA drugs unising pretrained models

```
from PSE import utils, model

# return the FDA approved drug dictionary
fda_drug = utils.fda_drugs_name(all)

# extracts the corespondin Gene-Side effect (GSE) graphs
g1 = utils.gse_graph(fda_drugs['drug1'])
g2 = utils.gse_graph(fda_drugs['drug2'])
g3 = utils.gse_graph(fda_drugs['drug3'])

# loads the pretraoned SiGCN model
pse_model = model.load_sigcn()





```
