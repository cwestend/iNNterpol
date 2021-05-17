# iNNterpol
Neural Network Interpolator for Atlas9 stellar model atmospheres

The aim is to obtain a fast and reliable way to recover the model atmospheres calculated for the ATLAS9 model data for the 
APOGEE sky survey (see http://research.iac.es/proyecto/ATLAS-APOGEE//) for the 5-dimensional grid they were calculated (dimensions 
being the  metalicity M/H, carbon abundance C/M, other-element abundance O/M, Effective Temperature Teff and Gravity (log)). 
The iNNterpol not only recovers the four atmospheric parameters of mass, temperature, pressure and electronic density at the given 
grid points but also for intermediate values of these.

Using the  ATLAS9 model data we trained a deep neural-network based on a previous 12-component Singular-Value Decomposition (SVD) of these models.

The SVD guarantees a continuity in optical depth while reduced the dimensionality of the problem. A fully connected Neural Network (NN), was trained
on the SVD coefficients. A 12-layer deep NN, with 40 nodes on each layer showed to give optimal results as shown on the recovered errors
and the fluxes (line profiles) calculated from these recovered atmospheres.

The code is provided together with the SVD coefficients in order to be able to rapidly reconstruct each model atmosphere. 
It also provides an extremely fast way of interpolating models between the values established. Models for areas where no models exist 
can also be recovered, a study of these shall be undertaken in a future paper.

# Requirements:


```

Python 3.6/3.8 with Pytorch 11.1.


```
# Data:

Download the coefficients of the 12 component SVD from a [python npz compressed file](https://cloud.iac.es/index.php/s/oNjrKkPHjn42fbe). 
