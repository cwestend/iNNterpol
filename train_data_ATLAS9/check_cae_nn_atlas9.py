# -*- coding: utf-8 -*-
"""

Atlas 9

Program to recreate the index arrays and obtaint those used for training, for
validating and for test (or check). The latter were never seen by the NN.


Usage: 
arr_ind_check, value_index, arr_ind_check_d = check_nn()

"""

import numpy as np
import matplotlib.pyplot as pl
import sys
import time


def check_nn(check=True):

    try:
        # index files
        ind_data=np.load('./atlas9_cae_nn_check_index.npz', allow_pickle=True)
        print('Reading index files')
    except:
        print('Cannot find the index file')
        sys.exit()

    try:
        # coefficients file
        cov_data=np.load('./atlas9_covs_ind_coefflogs12.npz', allow_pickle=True)
        #cov_data=np.load('../atlas9_covs_ind_coefflogs9.npz', allow_pickle=True)
    except:
        print('Cannot find the index file')
        sys.exit()

    # Indexes of the un-flattened array with non-null model values
    ind_orig = cov_data['ind']
    ind = ind_orig[0]

    # Read indexes
    if check:
        print('Using check indexes (untranided data)')
        ind_chk=ind_data['check_index']
    else:
        print('Using validation indexes (trainded data)')
        ind_chk=ind_data['train_index']
        
    """
    # Indices para mapear los valores de ind_chk en el array reconstruido con ceros
    ind_orig = sli_data['ind']
    ind = ind_orig[0]

    # No hace falta calcular, los leo
    all_temp = sli_data['all_temp']
    all_logg = sli_data['all_logg']
    metal_m  = sli_data['metal_m']
    carbon_m = sli_data['carbon_m']
    other_m = sli_data['other_m']

    value_index = sli_data['value_index']
    """

    # Array of all possible values of temp
    all_temp = np.append(np.arange(3500.,12000., 250),
            np.append(np.arange(12000.,20000., 500),np.arange(20000.,31000., 1000)))
    # Array of all possible values of logg
    all_logg = np.arange(0., 51., 5)
    # Arrays of values of metalicity (log)
    metal_m = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.75, -2.5, -2.25, -2.0, -1.75,
            -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0,
            1.5]

    carbon_m = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    other_m = [-1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    # Array dimensions 
    len_temp  = len(all_temp)
    len_logg  = len(all_logg)
    len_metal = len(metal_m)
    len_carbon = len(carbon_m)
    len_other = len(other_m)  

    # Array of indexes
    mm,cc,oo,tt,gg = np.meshgrid(np.arange(len_metal),np.arange(len_carbon),
                     np.arange(len_other),np.arange(len_temp),np.arange(len_logg), indexing='ij')

    # Array that associates the flattened array with those of non-null model values in the un-flattened one 
    value_index = np.vstack((mm.ravel(),cc.ravel(), oo.ravel(), tt.ravel(), gg.ravel())).T

    arr_ind_check_d = dict()
    arr_ind_check = np.empty((len(ind_chk),6),dtype=object)

    print('Reconstructing the arrays and dictionary of indexes...')
    tic = time.perf_counter()

    for i in range(len(ind_chk)):
        # Index within the un-flattened array
        indice = ind[ind_chk[i]]
        # Index in slicelogn (flattened arrays where n = mass, temp, pres, elect. dens)
        indice_slice = ind_chk[i]
        met_val = metal_m[value_index[indice][0]]
        car_val = carbon_m[value_index[indice][1]]
        oth_val = other_m[value_index[indice][2]]
        tem_val = all_temp[value_index[indice][3]]
        log_val = all_logg[value_index[indice][4]]
    
        #met_ind = value_index[indice][0]
        #car_ind = value_index[indice][1]
        #oth_ind = value_index[indice][2]
        #tem_ind = value_index[indice][3]
        #log_ind = value_index[indice][4]

        #arr_ind_check[i,...] = indice, met_val, car_val, oth_val, tem_val, log_val
        #print(met_val, car_val, oth_val, tem_val, log_val)
        arr_ind_check_d[(met_val, car_val, oth_val, tem_val, log_val)] = indice_slice
        arr_ind_check[i] = np.array([indice_slice, met_val, car_val, oth_val, tem_val, log_val])

    toc = time.perf_counter()
    print(f"Reconstructed in {toc - tic:0.4f} secs")

    return arr_ind_check, value_index, arr_ind_check_d
