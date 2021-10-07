# -*- coding: utf-8 -*-
"""

Atlas 9

Define atlas9_inter_cae_nn() function to interpolate in values on a grid of model atmospheres
based on the Atlas9 dataset calculated for the APOGEE sky survey
(http://research.iac.es/proyecto/ATLAS-APOGEE/).

Uses a trained Neural Network (16 layers, fully connected) over the resulting encoding using a CAE with
a bottleneck of 71 elements. extracted over all the data.

Uses the hyperparameter file in .pth format (pyTorch) of both the NN and the CAE, together with the 
model.py containing the training model and model_encoder.py containing the model CAE. 

Autor: cwestend (https://github.com/cwestend/iNNterpol)

"""

import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.utils.data

import glob
import os
import sys
import time

# Need scipy to work in np.longdouble format
#from scipy import linalg

# If linearly interpolating
# use https://github.com/cwestend/iNNterpol_PCA12/iNNterpol_PCA12.py

try:
    # hyperparam file 
    files = glob.glob("./*.pth")
    print('Reading the NN hyperparameter files: %s' % files[0])
    files_enc = glob.glob("./*.pth_encoder")
    print('Reading the CAE hyperparameter files: %s' % files_enc[0])
except:
    print('Cannot find the hyperparameter files')
    sys.exit()
"""
try:
    # SVD coefficients/covariances file 
    file_cov = glob.glob("./atlas9_covs_ind_coefflogs12.npz")
    
    cov_data = np.load(file_cov[0], allow_pickle=True)
    print('Reading the SVD file: %s' % file_cov[0])
except:
    print('Cannot find the SVD file')
    sys.exit()
"""

def parse_input(in_values):
    try:
        #in_values = np.array(in_values, dtype=float32) 
        in_val = np.copy(np.array(in_values, dtype=float))
        if len(in_val) != 5:
            raise ValueError("Incorrect size of input parameters")
        else:
            # Logg in 10x format
            in_val[4] = 10*in_val[4]
            return in_val
    except:
        raise ValueError("Need an array-like with 5 params: metalicity(log), carbon(log), other(log), temp, g(log)")

def innterpol(input_values):
    """ Function to interpolate in Atlas9 data: input 5 values of metal, carbon, other, temp, 10xlogg
        (array like) and outputs an array(71,4) with optical depth variation of mass(log), temp(log), 
        pres(log), and electronic density(log)"""

    try:
        # Import the model form local: there should be a model.py in dir
        import model
        import model_encoder
    except:
        print('Cannot find the model files (model.py and model_encoder.py) in current directory')
        sys.exit()

    inp_val = parse_input(input_values)
    # Rescale as trained (so they had similar size)
    inp_val[0] = inp_val[0]/5.
    inp_val[3] = inp_val[3]/30000.
    inp_val[4] = inp_val[4]/50. 


    # Values for de-normalizing
    # obtained for each parameter by (i.e for temp):
    # maxt_all = slicelogt.max(axis=1, keepdims=True)
    # mint_all = slicelogt.min(axis=1, keepdims=True)
    # maxt = round(maxt_all.mean(),2)
    # mint = round(mint_all.mean(),2)  
    maxm = 1.43
    minm = -4.7
    maxt = 4.35
    mint = 3.75
    maxp = 4.68
    minp = -1.41
    maxe = 15.44 
    mine = 8.61

    # The interpolating NN
    device = "cpu"
    checkpoint = max(files, key=os.path.getctime)
    chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
    hyperparameters = chk['hyperparameters']
    model = model.Network(hyperparameters).to(device)
    # model = model.MLP(hyperparameters).to(device)
    model.load_state_dict(chk['state_dict'])
    model.eval()

    # The CAE that encodes/decodes (model with CAE weights for encoding/decoding)
    checkpoint_enc = max(files_enc, key=os.path.getctime)
    chk_enc = torch.load(checkpoint_enc, map_location=lambda storage, loc: storage)
    hyperparameters_enc = chk_enc['hyperparameters']
    model_encoder = model_encoder.Network(hyperparameters_enc).to(device)
    model_encoder.load_state_dict(chk_enc['state_dict'])
    model_encoder.eval()
    

    # Output of the NN
    # We only need the forward pass, so we do not accumulate gradients
    with torch.no_grad():
        global out_nn
        # We transform the input from Numpy to PyTorch tensor
        inputs = torch.tensor(inp_val.astype('float32')).to(device)
        out_nn = model(inputs)

    # Format to apply decoder (and recover the parameter stratification)
    out_nn_t = out_nn.unsqueeze(1).unsqueeze(0)

    # Applying the decoder on the predicted values by the NN
    # We only need the forward pass, so we do not accumulate gradients
    with torch.no_grad():
        global out_de
        inputs = out_nn_t.to(device)
        out_de = model_encoder.decoder(inputs)

    # Convert to array
    # We bring the result to the CPU (if it was on the GPU) and transform to Numpy
    out_de_n = out_de.cpu().numpy()

    # Transform to m,t,p and e parameters in depth
    atlas9_cae_nn = np.empty((71, 4),dtype=object)  

    atlas9_cae_nn[:,0] = out_de_n[0,0,:]*(maxm-minm)+minm
    atlas9_cae_nn[:,1] = out_de_n[0,1,:]*(maxt-mint)+mint
    atlas9_cae_nn[:,2] = out_de_n[0,2,:]*(maxp-minp)+minp
    atlas9_cae_nn[:,3] = out_de_n[0,3,:]*(maxe-mine)+mine


    return atlas9_cae_nn