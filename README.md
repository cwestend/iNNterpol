# iNNterpol
### Neural Network Interpolator for Atlas9 stellar model atmospheres

AThe aim is to obtain a fast and reliable way to recover the model atmospheres calculated for the ATLAS9 model data for the 
APOGEE sky survey (see http://research.iac.es/proyecto/ATLAS-APOGEE//). These atmospheres were calculated
for a 5-dimensional grid, dimensions being the  metalicity M/H, carbon abundance C/M, other-element abundance O/M, effective 
Temperature and (log) surface Gravity. The iNNterpol not only recovers the four atmospheric parameters of mass, temperature, 
pressure and electronic density at the grid points they were calculated for but also for intermediate values of these.

Using the  ATLAS9 model data we trained a deep neural-network based on a previous 12-component Principal component analysis (PCA) 
obtained by Singular-Value Decomposition (SVD) analysis of these models.

The SVD guarantees a continuity in optical depth while reduced the dimensionality of the problem. A fully connected Neural Network (NN), was trained
on the 12 PCA coefficients considered. A 12-layer deep NN, with 40 nodes on each layer showed to give optimal results as shown on the recovered errors
and the fluxes (line profiles) calculated from these recovered atmospheres.

The code is provided together with the PCA coefficients in order to be able to rapidly reconstruct each model atmosphere. 
It also provides an extremely fast way of interpolating models between the calculated values. Models for areas (or grid values) where no data
is available on one or more of the extremes of the grid values can also be recovered. These regions can be considered as 
unbounded and thus a linear interpolation cannot be applied. Our iNNterpol method is able to learn from the data and give 
a smooth and reasonable values on these regions provided a reasonable departure in distance from grid values that do have 
calculated model data. A study of these regions, together with the quality of the recovered models and the fluxes obtained 
from these shall be undertaken in a future paper.

# Requirements:


```

Python 3.6/3.8 with Pytorch 11.1 (possibly 10.2 is ok)


```
## Data:

Download the coefficients of the 12 component PCA/SVD from a [python .npz compressed file](https://cloud.iac.es/index.php/s/oNjrKkPHjn42fbe). 

## Files:

Together with the above coefficient file, a model.py with the training model used is needed aswell as a *.pth file with the hyperparameters resulting from the NN training.

```
atlas9_covs_ind_coefflogs12.npz
model.py
<date-time>.pth

```

## Usage

Just run it in the directory with the above files, and you can get the interpolated atmosphere in (log) mass, temp, pressure and electronic density (indexes from 0 to 3). For example, to get the temperature for M/H = 0, C/M=O/M=0, Teff = 5000 and logg = 2.5 you can do in an interactive python shell:

```
% run "./innterpol.py"

% nn_innterp = innterpol([0, 0, 0, 5500, 2.5])
```

and you can plot it out rebuilding the optical depths as in:

```
% taur = np.arange(-6.875, 2, 0.125)

% import matplotlib.pyplot as pl 
% pl.plot(taur, 10**nn_innterp[:,0])


```

and for the same price, you get a **linear interpolator** that when used on the grid values recovers the Atlas9 model atmosphere quite precisely (to less than 0.2% (rms) accuracy):

```
% nn_model_lin = innterpol([0, 0, 0, 5500, 2.5], linear=True)

```


