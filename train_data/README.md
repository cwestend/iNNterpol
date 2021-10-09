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
% arr_ind_check, value_index, arr_ind_check_d = check_nn(check=False)
```


