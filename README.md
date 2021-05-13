# iNNterpol
Neural Network Interpolator for Atlas9 stellar model atmospheres

Using ATLAS9 the model atmospheres calculated for the for the APOGEE sky survey
(see http://research.iac.es/proyecto/ATLAS-APOGEE//) we trained a deep neural-network based on a previous 12-component SVD of these models.

The SVD guaranteed a continuity in optical depth and reduced the dimensionality of the problem. A 12-layer, fully connected NN, with 40 nodes on each layer showed to give optimal results.

The code is provided toguether with the SVD coefficients in order to be able to rapidly reconstruct each model atmosphere (in its 4 parameters of mass, temperature, pressure and electronic density). It also provides an extremely fast way of interpolating models between the values established. Models for areas where no models exist can also be recovered, a study of these shall be undertaken in a future paper.

# Requirements:


```

Python 3.6/3.8 with Pytorch 11.1.


```
