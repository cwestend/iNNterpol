# iNNterpol - train data
### Neural Network Interpolator for Atlas9 stellar model atmospheres (12 component PCA and 71 element CAE)


This is the data used to train both the CAE and the NN for https://github.com/cwestend/iNNterpol


# Requirements:


```

Python 3.6/3.8 


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

