# iNNterpol - training of the CAE (MARCS model data)
### Neural Network Interpolator for MARCS stellar model atmospheres (71 element CAE)


This is the data used to train the CAE that will be used in a NN for https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_MARCS


# Requirements:


```

Python 3.8/3.9 with Cuda 11.1 (possibly 10.2 is ok)


```
## Data:

The full MARCS dataset is needed to train the CAE. Please see Acknowledgements below if using this data.

The data is the full MARCS grid, in log10 and np.longdouble format for precision. It is split up into four slices, one
for each physical parameter (mass, temp, press, electron density). It is available as a
[numpy .npz compressed file](https://cloud.iac.es/index.php/s/joottHbXarQDALs). 

## Files:

Together with the above data file, a model_cuatro.py that contains the CAE and a train_marcs_cae.py that trains 
the NN minimizing the loss between the output and the input. The four physical parameters of each model 
are input as different channels into the CAE to capture the non-linearities in the data. 

```
train_marcs_cae.py 

model_cuatro.py

```

## Acknowledgements:

If using ATLAS9 or MARCS model data please cite:   
NEW ATLAS9 AND MARCS MODEL ATMOSPHERE GRIDS FOR THE APACHE POINT OBSERVATORY GALACTIC EVOLUTION EXPERIMENT (APOGEE).  
 Sz. Mészáros1,2, C. Allende Prieto1,2, B. Edvardsson3, F. Castelli4, A. E. García Pérez5, B. Gustafsson3, S. R. Majewski5, B. Plez6, R. Schiavon7, M. Shetrone8, and A. de Vicente1,2  
1 Instituto de Astrofísica de Canarias (IAC), E-38200 La Laguna, Tenerife, Spain  
2 Departamento de Astrofísica, Universidad de La Laguna (ULL), E-38206 La Laguna, Tenerife, Spain  
3 Department of Physics and Astronomy, Division of Astronomy and Space Physics, Box 515, SE-751 20 Uppsala, Sweden  
4 Istituto Nazionale di Astrofisica, Osservatorio Astronomico di Trieste, via Tiepolo 11, I-34143 Trieste, Italy  
5 Department of Astronomy, University of Virginia, P.O. Box 400325, Charlottesville, VA 22904-4325, USA  
6 Laboratoire Univers et Particules de Montpellier, Université Montpellier 2, CNRS, F-34095 Montpellier, France  
7 Gemini Observatory, 670 North A'ohoku Place, Hilo, HI 96720, USA  
8 McDonald Observatory, University of Texas, Austin, TX 78712, USA  

ADS: https://ui.adsabs.harvard.edu/abs/2012AJ....144..120M/abstract

DOI: 10.1088/0004-6256/144/4/120

## Training

Run the train_marcs_cae.py in the directory with the above files to train the CAE:

```
% python ./train_marcs_cae.py --epochs 50 --batch 32

```

This will end after the specified epochs, the weigths are in the weights_marcs_cae_71/ directory
with a *.pth extention.

These *.pth weights together with the model is used in the NN later on to create the CAE-NN interpolator 
iNNterpol. 

NOTE: The already trained weights for the CAE (*.pth_encoder*) can be downloaded 
from https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_MARCS



