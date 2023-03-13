# iNNterpol - train data
### Neural Network Interpolator for Atlas9 stellar model atmospheres (12 component PCA and 71 element CAE)


This is the data used to train both the CAE and the NN for https://github.com/cwestend/iNNterpol


# Requirements:


```

Python 3.6/3.8 with Pytorch 11.1 (possibly 10.2 is ok)


```
## Data:

Download the coefficients of the 12 component PCA/SVD from a [python .npz compressed file](https://cloud.iac.es/index.php/s/oNjrKkPHjn42fbe).

Download the indexes used for training, validating and testing (check) in the NN in a [python .npz compressed file](https://cloud.iac.es/index.php/s/XokeDEQ3eHowtwZ)..

(optional) Download the full Atlas9 dataset (2.2Gb!) by a [python .npz compressed file](https://cloud.iac.es/index.php/s/aEBE2dAao4Wc6JF). 

## Files:

Together with the above coefficients file, a check_cae_nn_atlas9.py that reconstructs the indexes used for training, for
validating and for test (or check). The latter were never seen by the NN.

```
atlas9_covs_ind_coefflogs12.npz

atlas9_cae_nn_check_index.npz

check_cae_nn_atlas9.py 

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

## Usage

Run the check_cae_nn_atlas9.py in the directory with the above files to get the indexes needed to reconstruct the models used for checking (test):

```
% run "./check_cae_nn_atlas9.py"

% arr_ind_check, value_index, arr_ind_check_d = check_nn()
```

to get the validation indexes instead:

```
% arr_ind_valid, value_index, arr_ind_valid_d = check_nn(check=False)
```

These indexes are key to recovering the models used for test/validating. For instance, to recover the parameters for the model number 12:

```
% arr_ind_check[11]
% array([584778.0, -0.25, -1.5, 0.75, 4500.0, 45.0], dtype=object)
```
where the first number is the index of the flattened array that can be obtained from the actual data (full model set file atlas9_nn_slicelog_index.npz).
The rest are the parameters: [M/H], [C/M], [O/M], Teff and 10*logg.
To get the 71 points in depth of the temperature model:

```
% data_slicelog = np.load('./atlas9_nn_slicelog_index.npz')
% slicelogt = data_slicelog['slicelogt']

% y_temp_model = slicelogt[int(arr_ind_check[11][0]),:]
```

This can be directly used for comparison with the model recovered by iNNterpol (see https://github.com/cwestend/iNNterpol):


```
% run "./innterpol.py"

% inp_val =  np.array((arr_ind_check[11][1], arr_ind_check[11][2], arr_ind_check[11][3],
                       arr_ind_check[11][4], arr_ind_check[11][5]/10.))
% nn_innterp = innterpol(inp_val)

```
Both plots are compared in the plot below:

```
% import matplotlib.pyplot as pl

% pl.plot(taur, y_temp_model, label=r'$Atlas9\; 4500K,\: [M/H] = -0.25,\: [C/M] = -1.5,\: [\alpha/M] = 0.75,\: logg = 4.5$')
% pl.plot(taur, nn_innterp[:,1], '-.', label=r'$iNNterpol\; 4500K,\: [M/H] = -0.25,\: [C/M] = -1.5,\: [\alpha/M] = 0.75,\: logg = 4.5$')

```

![Temperature plot of original and obtained untrained (test) atmosphere](../../assets/Temp_CAE_16_71_Teff4500_lgg45_github.png?raw=true)
