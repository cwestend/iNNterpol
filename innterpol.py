# -*- coding: utf-8 -*-
"""

Atlas 9

Define atlas9_inter_nn() function to interpolate in values on a grid of model atmospheres
based on the Atlas9 dataset calculated for the APOGEE sky survey
(http://research.iac.es/proyecto/ATLAS-APOGEE/).

Uses a trained Neural Network (12 layers, fully connected) of a 12 component SVD base 
extracted over all the data.

Uses the hyperparameter file in .pth format (pyTorch), the numpy savez file with the
covariances and needs a model.py with the training model. 

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
from scipy import linalg

# If linearly interpolating
from scipy.interpolate import RegularGridInterpolator 

coeffm9reb = np.empty([0])
coefft9reb = np.empty([0])
coeffp9reb = np.empty([0])
coeffe9reb = np.empty([0])
interpolm = RegularGridInterpolator
interpolt = RegularGridInterpolator
interpolp = RegularGridInterpolator
interpole = RegularGridInterpolator

try:
    # hyperparam file 
    files = glob.glob("./*.pth")
    print('Reading the hyperparameter file: %s' % files[0])
except:
    print('Cannot find the hyperparameter file')
    sys.exit()

try:
    # SVD coefficients/covariances file 
    file_cov = glob.glob("./atlas9_covs_ind_coefflogs12.npz")
    
    cov_data = np.load(file_cov[0], allow_pickle=True)
    print('Reading the SVD file: %s' % file_cov[0])
except:
    print('Cannot find the SVD file')
    sys.exit()

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

def initialize(coeffm9reb, coefft9reb, coeffp9reb, coeffe9reb, interpolm, interpolt, interpolp, interpole, coefs):
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
    len_todas = len_temp * len_logg * len_metal * len_carbon * len_other
    # If variables exist re-use them (only read once)
    if (len(coeffm9reb) == 0 and len(coefft9reb) == 0 and
        len(coeffp9reb) == 0 and len(coeffe9reb) == 0):
        # Coefficients for linear interpolation
        print('Reading SVD coefficients for linear interpolation')
        tic = time.perf_counter()
        coeffm9=cov_data['coefflogm9']
        coefft9=cov_data['coefflogt9']
        coeffp9=cov_data['coefflogp9']
        coeffe9=cov_data['coeffloge9']
        toc = time.perf_counter()
        print(f"Read in {toc - tic:0.4f} secs.")

        # Indexes of non null values    
        ind = cov_data['ind']
        print('Rebuilding coefficient matrices')
        coeffm9reb = np.empty((1786202, coefs), dtype=object)  
        coefft9reb = np.empty((1786202, coefs), dtype=object)
        coeffp9reb = np.empty((1786202, coefs), dtype=object)
        coeffe9reb = np.empty((1786202, coefs), dtype=object)
        coeffm9reb[ind[0],:] = coeffm9  
        coefft9reb[ind[0],:] = coefft9  
        coeffp9reb[ind[0],:] = coeffp9  
        coeffe9reb[ind[0],:] = coeffe9   
        coeffm9reb = coeffm9reb.reshape(len_metal, len_carbon, len_other, 
                                len_temp, len_logg, coefs) 
        coefft9reb = coefft9reb.reshape(len_metal, len_carbon, len_other, 
                                len_temp, len_logg, coefs) 

        coeffp9reb = coeffp9reb.reshape(len_metal, len_carbon, len_other, 
                                len_temp, len_logg, coefs) 
        coeffe9reb = coeffe9reb.reshape(len_metal, len_carbon, len_other, 
                                len_temp, len_logg, coefs) 
        print('Creating interpolator functions')

        interpolm = RegularGridInterpolator((metal_m,carbon_m,other_m, all_temp, 
                                all_logg,np.arange(coefs)),coeffm9reb,bounds_error=True)
        interpolt = RegularGridInterpolator((metal_m,carbon_m,other_m, all_temp, 
                                all_logg,np.arange(coefs)),coefft9reb,bounds_error=True)
        interpolp = RegularGridInterpolator((metal_m,carbon_m,other_m, all_temp, 
                                all_logg,np.arange(coefs)),coeffp9reb,bounds_error=True)
        interpole = RegularGridInterpolator((metal_m,carbon_m,other_m, all_temp, 
                                all_logg,np.arange(coefs)),coeffe9reb,bounds_error=True)

    return interpolm, interpolt, interpolp, interpole, coeffm9reb, coefft9reb, coeffp9reb, coeffe9reb

def innterpol(input_values, linear=False, verbose=True, coef12=True):
    """ Function to interpolate in Atlas9 data: input 5 values of metal, carbon, other, temp, 10xlogg
        (array like) and outputs an array(71,4) with optical depth variation of mass(log), temp(log), 
        pres(log), and electronic density(log)"""

    try:
        # Import the model form local: there should be a model.py in dir
        import model
    except:
        print('Cannot find the model file (model.py) in current directory')
        sys.exit()

    device = "cpu"
    
    global coeffm9reb
    global coefft9reb
    global coeffp9reb
    global coeffe9reb
    global interpolm
    global interpolt
    global interpolp
    global interpole

    # Covariance matrixes
    covmlog=cov_data['covm']
    covtlog=cov_data['covt']
    covplog=cov_data['covp']
    covelog=cov_data['cove']

    # Calculate bases
    Umlog, Wm, Vm = linalg.svd(covmlog)
    Utlog, Wt, Vt = linalg.svd(covtlog)
    Uplog, Wp, Vp = linalg.svd(covplog)
    Uelog, We, Ve = linalg.svd(covelog)

    # Space for other bases (9/12 components)
    if coef12:
        coefs = 12
    else:
        coefs = 9    
    
    new_basem = Umlog[:,:coefs]
    new_baset = Utlog[:,:coefs]
    new_basep = Uplog[:,:coefs]
    new_basee = Uelog[:,:coefs]

    if linear:
        interpolm, interpolt, interpolp, interpole, coeffm9reb, coefft9reb, coeffp9reb, coeffe9reb = initialize(coeffm9reb, coefft9reb, coeffp9reb, coeffe9reb, interpolm, interpolt, interpolp, interpole, coefs)

        # Transform to m,t,p and e parameters in depth
        input = parse_input(input_values)

        if verbose:
            print('Interpolating on coefficients and rebuilding atmospheric params.')
        atlas9_nn = np.empty((71, 4),dtype=object)
        coefim = []
        coefit = []
        coefip = []
        coefie = []
        try:
            for i in range(coefs):
                aa = ([input[0], input[1], input[2], input[3], input[4], i])
                coefim.append(interpolm(aa))
                coefit.append(interpolt(aa))
                coefip.append(interpolp(aa))
                coefie.append(interpole(aa))

            coefim_ = np.array(coefim)[:,0]
            coefit_ = np.array(coefit)[:,0]
            coefip_ = np.array(coefip)[:,0]
            coefie_ = np.array(coefie)[:,0]

            atlas9_nn[:,0] = coefim_ @ new_basem.T
            atlas9_nn[:,1] = coefit_ @ new_baset.T
            atlas9_nn[:,2] = coefip_ @ new_basep.T
            atlas9_nn[:,3] = coefie_ @ new_basee.T
        except ValueError:
            print('Cannot interpolate: some value is out of bounds (would extrapolate)')
        except:
            raise
    else:
        # Use NN for interpolation
        checkpoint = max(files, key=os.path.getctime)

        if verbose:
            print("=> loading checkpoint '{}'".format(checkpoint))
        chk = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        if verbose:
            print("=> loaded checkpoint '{}'".format(checkpoint))

        # Build the network again using the hyperparameters
        hyperparameters = chk['hyperparameters']
        model = model.Network(hyperparameters).to(device)

        # Now we load the weights and the network should be on the same state as before
        model.load_state_dict(chk['state_dict'])

        # Put the network in evaluation mode
        model.eval()

        inp = parse_input(input_values)
        # Rescale as trained (so they had similar size)
        inp[0] = inp[0]/5.
        inp[3] = inp[3]/30000.
        inp[4] = inp[4]/50.

        # We only need the forward pass, so we do not accumulate gradients
        with torch.no_grad():
            global out
            
            # We transform the input from Numpy to PyTorch tensor
            #inputs = torch.tensor(y.astype('float32')).to(self.device)
            inputs = torch.tensor(inp.astype('float32')).to(device)
            # And call the model
            out = model(inputs)
        # We bring the result to the CPU (if it was on the GPU) and transform to Numpy
        out = out.cpu().numpy()

        # Transform to m,t,p and e parameters in depth
        atlas9_nn = np.empty((71, 4),dtype=object)
        atlas9_nn[:,0] = out[:coefs] @ new_basem.T
        atlas9_nn[:,1] = out[coefs:2*coefs] @ new_baset.T
        atlas9_nn[:,2] = out[2*coefs:3*coefs] @ new_basep.T
        atlas9_nn[:,3] = out[3*coefs:4*coefs] @ new_basee.T

    return atlas9_nn