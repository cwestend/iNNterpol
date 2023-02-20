# -*- coding: utf-8 -*-
"""

PHOENIX 

Define phoenix_inter_cae_nn() function to interpolate in values on a grid of model atmospheres
based on the PHOENIX dataset as described in T.-O. Husser, S. Wende-von Berg, S. Dreizler, 
D. Homeier, A. Reiners, T. Barman and P. H. Hauschildt A&A 553, A6 (2013, https://doi.org/10.1051/0004-6361/201219058)

Uses a trained Neural Network (16 layers, fully connected) over the resulting encoding using a CAE with
a bottleneck of 96 elements. extracted over all the data.

Uses the hyperparameter file in .pth format (pyTorch) of both the NN and the CAE, together with the 
model.py containing the training model and model_encoder.py containing the model CAE. 

Autor: cwestend (https://github.com/cwestend/iNNterpol_PHOENIX)

"""

import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.utils.data

import glob
import os
import sys
import time


try:
    # hyperparam file 
    files = glob.glob("./*.pth")
    print('Reading the NN hyperparameter files: %s' % files[0])
    files_enc = glob.glob("./*.pth_encoder*")
    print('Reading the CAE hyperparameter files: %s' % files_enc[0])
except:
    print('Cannot find the hyperparameter files')
    sys.exit()

def parse_input(in_values):
    try:
        #in_values = np.array(in_values, dtype=float32) 
        in_val = np.copy(np.array(in_values, dtype=float))
        if len(in_val) != 4:
            raise ValueError("Incorrect size of input parameters")
        else:
        #    # Logg in 10x format
        #    in_val[4] = 10*in_val[4]
            return in_val
    except:
        raise ValueError("Need an array-like with 4 params: metalicity(log), carbon(log), Teff, g(log)")

def innterpol(input_values):
    """ Function to interpolate in PHOENIX data: input 4 values of metal, carbon, Teff, logg
        (array like) and outputs an array(64,4) with optical depth variation of mass(log), temp(log), 
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
    inp_val[0] = inp_val[0]/4.
    inp_val[2] = inp_val[2]/15000.
    inp_val[3] = inp_val[3]/6.5 

    # Values for de-normalizing
    # obtained for each parameter by (i.e for temp):
    # maxt_all = slicelogt.max(axis=1, keepdims=True)
    # mint_all = slicelogt.min(axis=1, keepdims=True)
    # maxt = round(maxt_all.mean(),2)
    # mint = round(mint_all.mean(),2)  
    maxm = 1.65
    minm = -9.26
    maxt = 4.03
    mint = 3.51
    maxp = 4.91
    minp = -6.0
    maxe = 14.88 
    mine = 3.25

    # Optical depth (if needed)
    # 64 points as read from Phoenix fits
    taur = np.array([0.0000e+00, 1.0000e-10, 1.5615e-10, 2.4384e-10, 3.8075e-10,
                5.9456e-10, 9.2841e-10, 1.4497e-09, 2.2638e-09, 3.5350e-09,
                5.5200e-09, 8.6195e-09, 1.3460e-08, 2.1017e-08, 3.2819e-08,
                5.1248e-08, 8.0025e-08, 1.2496e-07, 1.9513e-07, 3.0470e-07,
                4.7579e-07, 7.4296e-07, 1.1602e-06, 1.8116e-06, 2.8289e-06,
                4.4173e-06, 6.8978e-06, 1.0771e-05, 1.6819e-05, 2.6264e-05,
                4.1011e-05, 6.4040e-05, 1.0000e-04, 1.5615e-04, 2.4384e-04,
                3.8075e-04, 5.9456e-04, 9.2841e-04, 1.4497e-03, 2.2638e-03,
                3.5350e-03, 5.5200e-03, 8.6195e-03, 1.3460e-02, 2.1017e-02,
                3.2819e-02, 5.1248e-02, 8.0025e-02, 1.2496e-01, 1.9513e-01,
                3.0470e-01, 4.7579e-01, 7.4296e-01, 1.1602e+00, 1.8116e+00,
                2.8289e+00, 4.4173e+00, 6.8978e+00, 1.0771e+01, 1.6819e+01,
                2.6264e+01, 4.1011e+01, 6.4040e+01, 1.0000e+02])

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
    phoenix_inter_cae_nn = np.empty((64, 4),dtype=object)  

    phoenix_inter_cae_nn[:,0] = out_de_n[0,0,:]*(maxm-minm)+minm
    phoenix_inter_cae_nn[:,1] = out_de_n[0,1,:]*(maxt-mint)+mint
    phoenix_inter_cae_nn[:,2] = out_de_n[0,2,:]*(maxp-minp)+minp
    phoenix_inter_cae_nn[:,3] = out_de_n[0,3,:]*(maxe-mine)+mine


    return phoenix_inter_cae_nn, taur