# iNNterpol
Neural Network Interpolator for Atlas9 stellar model atmospheres

The aim is to obtain a fast and reliable way to recover the model atmospheres calculated for the ATLAS9 model data for the 
APOGEE sky survey (see http://research.iac.es/proyecto/ATLAS-APOGEE//). These atmospheres were calculated
for a 5-dimensional grid, dimensions being the  metalicity M/H, carbon abundance C/M, other-element abundance O/M, effective 
Temperature and (log) surface Gravity. The iNNterpol not only recovers the four atmospheric parameters of mass, temperature, 
pressure and electronic density at the grid points they were calculated for but also for intermediate values of these.

Using the  ATLAS9 model data we trained a deep neural-network based on a previous 12-component Singular-Value Decomposition (SVD) of these models.

The SVD guarantees a continuity in optical depth while reduced the dimensionality of the problem. A fully connected Neural Network (NN), was trained
on the SVD coefficients. A 12-layer deep NN, with 40 nodes on each layer showed to give optimal results as shown on the recovered errors
and the fluxes (line profiles) calculated from these recovered atmospheres.

The code is provided together with the SVD coefficients in order to be able to rapidly reconstruct each model atmosphere. 
It also provides an extremely fast way of interpolating models between the calculated values. Models for areas (or grid values) where no data
is available on one of the extremes can also be recovered. These regions can be considered as unbounded and thus a linear interpolation
cannot be applied. Our iNNterpol method is able to learn from the data and give a smooth and reasonable values on these regions provided a 
reasonable departure in distance from grid values that have calculated model data. A study of these regions, together with 
the quality of the recovered models and the fluxes obtained from these shall be undertaken in a future paper.

# Requirements:


```

Python 3.6/3.8 with Pytorch 11.1.


```
# Data:

Download the coefficients of the 12 component SVD from a [python .npz compressed file](https://cloud.iac.es/index.php/s/oNjrKkPHjn42fbe). 
