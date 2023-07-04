# iNNterpol - training of the NN with a CAE (MARCS model data)
### Neural Network Interpolator for MARCS stellar model atmospheres (NN with a 71 element CAE)

This is the final training of the NN to obrain the iNNterpol-MARCS. It needs the weights as explained in 
https://github.com/cwestend/iNNterpol/tree/main/training_MARCS/train_cae_MARCS/

**In press!** The paper is accepted and in press in Astronomy & Astrophisics, if you use the code, please cite it by its DOI: https://doi.org/10.1051/0004-6361/202346372. A pre-print version is available at https://arxiv.org/abs/2306.06938.

# Requirements:


```

Python 3.8/3.9 with Cuda 11.1 (possibly 10.2 is ok)


```
## Data:

The weights of the Convolutional Auto Encoder (CAE) trained over the MARCS dataset is needed to finally train the NN. 

Please train the CAE as explained in https://github.com/cwestend/iNNterpol/tree/main/training_MARCS/train_cae_MARCS/ or alternatively, 
use the already trained weights for the CAE (*.pth_encoder*) and model (which should be renamed to model_encoder.py)
from https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_MARCS

The full MARCS dataset is needed again to train the NN. Please see Acknowledgements in https://github.com/cwestend/iNNterpol/tree/main/training_MARCS/train_cae_MARCS/ if using this data.

The data is the full MARCS grid, in log10 and np.longdouble format for precision. It is split up into four slices, one
for each physical parameter (mass, temp, press, electron density). It is available as a
[numpy .npz compressed file](https://cloud.iac.es/index.php/s/joottHbXarQDALs). 

## Files:

Both the weights for the trained CAE and the model used for training has to be used and they have to be in the following format:

Additionally, the model.py containing the fully-connected Neural Network model must be used. The files should be named as in: 

```
train_marcs_cae_nn.py 

*.pth_encoder*

model_encoder.py

model.py

```

## Training

Run the train_marcs_cae_nn.py in the directory with the above files to train the NN with a CAE:

```
% python ./train_marcs_cae_nn.py --epochs 100 --batch 32

```

This finally trains the NN using the CAE as a dimensionality reduction.

After the specified epochs, the weigths of this final NN are in the weights_marcs_nn/ directory
with a *.pth extention.

These NN *.pth weights, the NN model (model.py), together with the CAE weights *.pth_encoder* and the CAE model (model_encoder.py)
are needed to be able to effectively interpolate in MARCS grid as described in https://github.com/cwestend/iNNterpol/tree/main/iNNterpol_MARCS

## Parameters

The important hyperparameters are in the code, but two have to be kept constant, mainly the 'input_size' as it is fixed due 
to the nature of the data (MARCS model grid uses metal, carbon, alpha, temp, logg as parameters) and 'output_size' which
has to be equal to the size of the CAE bottleneck (trained in https://github.com/cwestend/iNNterpol/tree/main/training_MARCS/train_cae_MARCS/).

in train_marcs_cae_nn.py:

```
# Hyperparameters for the neural network, see model.py
        hyperparameters = {
            # Number of input parameters: metal, carbon, alpha, temp, logg
            'input_size': 5,
            # Nodes each (hidden) fully-connected layer the NN has
            'hidden_size': 71,
            # Number of fully-connected layers of the NN
            'n_hidden_layers': 22,
            # Output - nodes the encoder has (must be identical to the CAE bottleneck!)
            'output_size': 71,
            # Activation function: pick you poison
            #'activation': 'relu'
            #'activation': 'leakyrelu'
            'activation': 'elu'
        }
```
