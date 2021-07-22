# GrasslandModels [![test-package](https://github.com/sdtaylor/GrasslandModels/workflows/test-package/badge.svg)](https://github.com/sdtaylor/GrasslandModels/actions)

A python implementation for Grassland Productivity/Phenology models.

- PhenoGrass model described in [Hufkens et al. 2016](http://www.nature.com/articles/nclimate2942).
- Pulse Repsonse models described in [Choler et al. 2011](https://doi.org/10.1007/s10021-010-9403-9) and [Choler et al. 2010](https://doi.org/10.5194/bg-7-907-2010)  

The PhenoGrass model was used in the following study:  
Taylor, S.D. and Browning, D.M., 2021. Multi-scale assessment of a grassland productivity model. Biogeosciences, 18(6), pp.2213-2220. https://doi.org/10.5194/bg-18-2213-2021. [Github Repo](https://github.com/sdtaylor/PhenograssReplication)  

## Installation
Requires: cython, scipy, pandas, joblib, and numpy

Install the latest version from Github  

```
pip install git+git://github.com/sdtaylor/GrasslandModels
```

## Usage  

The primary PhenoGrass model is described in `GrasslandModels/models/phenograss.py`, along with other models derived from the papers Choller et al. 2010,2011.  

Basic usage is as follows. This will load test data, initialize a PhenoGrass model instance, and fit the model to the data. 

```
from GrasslandModels import models, utils
import numpy as np

# Load some test phenocam data. GCC is a numpy array of shape (n_sites,n_timesteps)
# predictor_vars is a dict of the required drivers of the same shape as GCC.
# Except for site level variables (such as soil water holding capacity), which have the shape (n_sites,)

GCC, predictor_vars = utils.load_test_data()
model = models.Phenograss()
model.fit(observations=GCC, predictors=predictor_vars)
model.get_params()
{'b1': 29.622625671240186,
 'b2': 0.016353861309752915,
 'b3': 88.5852418465677,
 'b4': 3.056869299257741,
 'Phmax': 22.352299783003804,
 'Phmin': 7.2066838717002035,
 'Topt': 41.170453185881755,
 'L': 10.110308716318412,
 'h': 709.2796450222473}
```

The fitted PhenoGrass model from Hufkens et al. 2016 can also be loaded.

```
model = utils.load_prefit_model('PhenoGrass-original')
predictions = model.predict(predictor_vars)
# The absolute error
np.nanmean((predictions - GCC))
```
