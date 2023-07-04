# iNNterpol (PHOENIX)
### Neural Network Interpolator for PHOENIX stellar model atmospheres

![Temperature differences between original and obtained untrained (test) atmospheres](../../assets/DTemp_chkCAE_22_128_PHOENIX_10pct.png?raw=true)<br>
(Fig1. plot of Temperature differences in % between original and obtained for untrained (test) atmospheres of metal-rich dwarf stars.)

![Temperature differences in Kelvin between original and obtained untrained (test) atmospheres](../../assets/DTemp_Stats_chkCAE_16_96_PHOENIX_mp05_lgg45_scale.png?raw=true)<br>
(Fig1. plot of Temperature differences in K as above, with added RMS values)

**In press!** The paper is accepted and in press in Astronomy & Astrophisics, if you use the code, please cite it by its DOI: https://doi.org/10.1051/0004-6361/202346372. A pre-print version is available at https://arxiv.org/abs/2306.06938.

The aim is to obtain a fast and reliable way to recover the model atmospheres calculated for the PHOENIX dataset as described in 
T.-O. Husser, S. Wende-von Berg, S. Dreizler, D. Homeier, A. Reiners, T. Barman and P. H. Hauschildt A&A 553, A6 (2013, https://doi.org/10.1051/0004-6361/201219058). These atmospheres were calculated
for a 4-dimensional grid, dimensions being the  metalicity M/H, carbon abundance C/M, effective 
Temperature and (log) surface Gravity. The iNNterpol not only recovers the four atmospheric parameters of mass, temperature, 
pressure and electronic density at the grid points they were calculated for but also for intermediate values of these.

Using the PHOENIX model data we trained a deep neural-network based on a previous Convolutional Auto-Encoder (CAE)
that reduces the features of these models to a reduced parameter space (96 nodes). 

A 16-layer deep NN, with 50 nodes on each layer trained on the output of the 96 node CAE for all four input ([M/H], [Alpha/M], Teff. and logGravity) parameters simultaneously, showed to give optimal results as shown on the recovered errors
and the fluxes (line profiles) calculated from these recovered atmospheres.


The code is provided together with the trained weights of the CAE and NN in order to be able to rapidly reconstruct each model atmosphere. 

It also provides an extremely fast way of interpolating models between the calculated values. Models for areas (or grid values) where no data
is available on one or more of the extremes of the grid values can also be recovered. These regions can be considered as 
unbounded and thus a linear interpolation cannot be applied. Our iNNterpol method is able to learn from the data and give 
a smooth and reasonable values on these regions provided a reasonable departure in distance from grid values that do have 
calculated model data. A study of these regions, together with the quality of the recovered models and the fluxes obtained 
from these shall be undertaken in a future paper.

# Requirements:


```

Python 3.8 with Pytorch 11.12 (possibly 10.2 and above is ok)


```
## Data:

No need to download any data!. 

## Files:

The model.py with the training model for the NN and the model_encoder.py for the CAE encoder/decoder used are needed aswell as the *.pth files with the hyperparameters resulting from the NN and CAE training.

```

model.py
model_encoder.py
<date-time>.pth
<date-time-2>.pth_encoder

```

## Usage

Just run it in the directory with the above files, and you can get the interpolated atmosphere in (log) mass, temp, pressure and electronic density (indexes from 0 to 3). For example, to get the temperature for M/H = 0, Alpha/M=0, Teff = 5000 and logg = 2.5 you can do in an interactive python shell:

```
% run "./innterpol.py"

% nn_innterp, taur = innterpol([0, 0, 5500, 2.5])
```

and you can plot it out rebuilding the optical depths as in:

```

% import matplotlib.pyplot as pl 
% pl.xscale('log')
% pl.plot(taur, 10**nn_innterp[:,0])


```



