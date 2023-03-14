# -*- coding: utf-8 -*-
"""

MARCS data CAE train file

Trains the weights of a CAE on log10 data of MARCS model atmospheres.

The model is defined in model_cuatro.py where the four input layers (mass, temp, press and electron density)
of the MARCS model are the input (thus 56 steps in logtau) and the bottleneck of 71 is created as
the result of the encoder. The decoder is the inverse of the encoder and tries to reproduce the input
as clkose as possible.

The resulting weights in .pth format (pyTorch) are saved in the folder weights_marcs_cae_71/*.pth
These weights together with the model file (model_cuatro.py) are used in a fully-connected NN
to train the iNNterpol. 

Autors: cwestend (https://github.com/cwestend/iNNterpol) and aasesio (https://github.com/aasensio)

"""

import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.functional as tf
import torch.utils.data
import time
from tqdm import tqdm
import model_cuatro
import argparse

try:
    import nvidia_smi
    NVIDIA_SMI = True
except:
    NVIDIA_SMI = False
import sys
import os
import pathlib

import time

class Dataset(torch.utils.data.Dataset):
    """
    Dataset class that will provide data during training. Modify it accordingly
    for your dataset. 
    """
    def __init__(self, n_training):
        
        super(Dataset, self).__init__()

        # Enlace al fichero de saves

        # Save datafile with slices by log10 of each quantity (mass, temp, press, electron density) of all data.
        try:
            sli_data=np.load('./marcs_slicelog_index.npz', allow_pickle=True)
            print('Reading the file')
        except:
            print('Cannot find the save file with log10 of each quantity (mass, temp, press, electron density) of all data!')
            sys.exit()
        # Inicialize the arrays

        
        # Load the data
        print('Loading the data saves...')
        self.slicelogm = sli_data['slicelogm']
        self.slicelogt = sli_data['slicelogt']
        self.slicelogp = sli_data['slicelogp']
        self.sliceloge = sli_data['sliceloge']

        print('Normalizing the data...')
        maxm = self.slicelogm.max(axis=1, keepdims=True)
        minm = self.slicelogm.min(axis=1, keepdims=True)
        maxt = self.slicelogt.max(axis=1, keepdims=True)
        mint = self.slicelogt.min(axis=1, keepdims=True)
        maxp = self.slicelogp.max(axis=1, keepdims=True)
        minp = self.slicelogp.min(axis=1, keepdims=True)
        maxe = self.sliceloge.max(axis=1, keepdims=True)
        mine = self.sliceloge.min(axis=1, keepdims=True)
 
        # Use constant data thats easy to recover afterwards!
        maxm1 = round(maxm.mean(),2)
        minm1 = round(minm.mean(),2)
        maxt1 = round(maxt.mean(),2)
        mint1 = round(mint.mean(),2)
        maxp1 = round(maxp.mean(),2)
        minp1 = round(minp.mean(),2)
        maxe1 = round(maxe.mean(),2)
        mine1 = round(mine.mean(),2)

        self.slicelogm = (self.slicelogm - minm1)/(maxm1 - minm1)
        self.slicelogt = (self.slicelogt - mint1)/(maxt1 - mint1)
        self.slicelogp = (self.slicelogp - minp1)/(maxp1 - minp1)
        self.sliceloge = (self.sliceloge - mine1)/(maxe1 - mine1)

        # Index array with non null values (actual existent atmospheres)
        ind = sli_data['ind']   
        
        # All indexes of the data
        self.n_training = int(len(ind[0]))
        # 
        self.indices =  ind[0]

    def __getitem__(self, index):
        """
        Return each item in the training set

        Parameters
        ----------
        index : int
            index of element to return
        
        """
        
        # Input: array in 4 channels (m, t, p, ne) of 56 optical depth points for all models 
        #
        inp_rawm = self.slicelogm[index,:]
        inp_rawt = self.slicelogt[index,:]
        inp_rawp = self.slicelogp[index,:]
        inp_rawe = self.sliceloge[index,:]
        inp1dm = torch.tensor(list(map(float, inp_rawm)))
        inp1dt = torch.tensor(list(map(float, inp_rawt)))
        inp1dp = torch.tensor(list(map(float, inp_rawp)))
        inp1de = torch.tensor(list(map(float, inp_rawe)))
        inp = torch.cat((inp1dm,  inp1dt, inp1dp, inp1de)).reshape(4,56)


        # Output: array in 4 channels (m, t, p, ne) of 56 optical depth points for all models
        #
        out_rawm = self.slicelogm[index,:]
        out_rawt = self.slicelogt[index,:]
        out_rawp = self.slicelogp[index,:]
        out_rawe = self.sliceloge[index,:]
        out1dm = torch.tensor(list(map(float, out_rawm)))
        out1dt = torch.tensor(list(map(float, out_rawt)))
        out1dp = torch.tensor(list(map(float, out_rawp)))
        out1de = torch.tensor(list(map(float, out_rawe)))
        out = torch.cat((out1dm,  out1dt, out1dp, out1de)).reshape(4,56)
        
        return inp, out

    def __len__(self):
        return self.n_training        

class Training(object):
    """
    Main class to do the training
    
    """
    def __init__(self, batch_size, validation_split=0.2, check_split=0.10, gpu=0, smooth=0.05):

        # Check for the presence of a GPU
        self.cuda = torch.cuda.is_available()
        self.gpu = gpu        
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
        # Choose the GPU
        self.device = torch.device(2)

        # Smoothing factor for the loss
        self.smooth = smooth

        # If this package is installed, get information about the GPU
        if (NVIDIA_SMI):
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(self.gpu)
            print("Computing in {0} : {1}".format(self.device, nvidia_smi.nvmlDeviceGetName(self.handle)))
        
        # Batch size and how to split the training/validation
        self.batch_size = batch_size
        self.validation_split = validation_split  
        # Cuantos para comprobar (no entran en train/validate)
        self.check_split = check_split      
                
        kwargs = {'num_workers': 2, 'pin_memory': False} if self.cuda else {}        

        # Hyperparameters for the neural network, for info only (see model_cuatro.py)
        hyperparameters = {
            # Only parameter here is the activation function, which can be 'relu', 'leakyrelu' or 'elu'
            #'activation': 'relu'
            #'activation': 'leakyrelu'
            'activation': 'elu'
        }
        
        # Instantiate the model
        self.model_cuatro = model_cuatro.Network(hyperparameters).to(self.device)
        
        # Count the number of trainable parameters
        print('N. total parameters : {0}'.format(sum(p.numel() for p in self.model_cuatro.parameters() if p.requires_grad)))

        # Instantiate the dataset. We only use one dataset and then do the splitting training/validation
        self.dataset = Dataset(n_training=400000)

        print("n_training : {0}".format(self.dataset.n_training))
        
        # We then split the indices in training/validation
        # 80% train, 10% validate, 10% test/check 
        self.train_index = idx[0:int((1-validation_split)*self.dataset.n_training)]
        self.check_index = idx[int((1-validation_split)*self.dataset.n_training):int((1-validation_split+check_split)*self.dataset.n_training)]
        self.validation_index = idx[int((1-validation_split+check_split)*self.dataset.n_training):]
        
        """
        # Can be saved and recovered after, if consistency is needed.
        #np.savez_compressed('marcs_cae_index.npz.npz', check_index=self.check_index, validation_index=self.validation_index, train_index=self.train_index)
        # To read a previous save (80% - 10% - 10%)
        try:
            sli_data=np.load('marcs_cae_index.npz', allow_pickle=True)
            print('Reading the index files training/validating/test')
        except:
            print('Cannot find the save file with the indexes!')
            sys.exit()
        self.train_index = sli_data['train_index']
        self.check_index = sli_data['check_index']
        self.validation_index = sli_data['validation_index']
        """

        # Define samplers for the training and validation sets
        self.train_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.train_index)
        self.validation_sampler = torch.utils.data.sampler.SubsetRandomSampler(self.validation_index)
                
        # Data loaders that will inject data during training
        self.train_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, **kwargs)
        self.validation_loader = torch.utils.data.DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, sampler=self.validation_sampler, **kwargs)

    def init_optimize(self, epochs, lr, weight_decay, scheduler):
        """
        Initialize the optimization. Define optimizer, scheduler, loss, output files, etc.

        Parameters
        ----------
        epochs : int
            Number of epochs
        lr : float
            Learning rate
        weight_decay : float
            Weight decay (regularizer)
        scheduler : int
            Number of steps before applying learning rate scheduler
        """

        self.lr = lr
        self.weight_decay = weight_decay
        print('Learning rate : {0}'.format(lr))
        self.n_epochs = epochs
        
        # Create the output directory if it does not exist
        p = pathlib.Path('weights_marcs_cae_71/')
        p.mkdir(parents=True, exist_ok=True)

        # Define the output name, including date and time
        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = 'weights_marcs_cae_71/{0}'.format(current_time)

        # Copy model
        file = model_cuatro.__file__.split('/')[-1]
        shutil.copyfile(model_cuatro.__file__, '{0}_model.py'.format(self.out_name))
        shutil.copyfile('{0}/{1}'.format(os.path.dirname(os.path.abspath(__file__)), file), '{0}_trainer.py'.format(self.out_name))
        self.file_mode = 'w'
        
        # Optimizer (change according to your interest)
        self.optimizer = torch.optim.Adam(self.model_cuatro.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # Loss function  (change according to your interest)
        self.loss_fn = nn.MSELoss().to(self.device)

        # Scheduler (change according to your interest)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=scheduler, gamma=0.5)

    def optimize(self):
        """
        Do the real work
        """

        # Lists that will store the training/validation losses
        self.train_loss = []
        self.valid_loss = []
        best_loss = float('inf')
        
        print('Model : {0}'.format(self.out_name))

        # Loop on epochs
        for epoch in range(1, self.n_epochs + 1):            

            # Do one epoch for training and validation
            train_loss = self.train(epoch)
            valid_loss = self.validate(epoch)

            # Store current losses
            self.train_loss.append(train_loss)
            self.valid_loss.append(valid_loss)

            # Scheduler step
            self.scheduler.step()

            # Save results if the validation loss is the best
            if (valid_loss < best_loss):
                best_loss = valid_loss
                
                checkpoint = {
                    'epoch': epoch + 1,
                    'state_dict': self.model_cuatro.state_dict(),
                    'train_loss': self.train_loss,
                    'valid_loss': self.valid_loss,
                    'best_loss': best_loss,
                    'hyperparameters': self.model_cuatro.hyperparameters,
                    'optimizer': self.optimizer.state_dict(),
                }
                
                print("Saving model...")
                torch.save(checkpoint, f'{self.out_name}.pth')
        
    def train(self, epoch):
        """
        Training

        Parameters
        ----------
        epoch : int
            Epoch

        """

        # Put the model in training mode
        self.model_cuatro.train()
        
        print("Epoch {0}/{1}".format(epoch, self.n_epochs))

        # Iterator for the training set, including progress bar
        t = tqdm(self.train_loader)
        loss_avg = 0.0
        n = 1
        
        # Get current learning rate from the optimizer
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        # Loop over the training set in batches
        for batch_idx, (inputs, outputs) in enumerate(t):

            # Move to the computing device (CPU or GPU)
            inputs = inputs.to(self.device)
            outputs = outputs.to(self.device)
            
            # Reset optimizer gradients
            self.optimizer.zero_grad()

            # Evaluate model
            out = self.model_cuatro(inputs)
            
            # Compute loss
            loss = self.loss_fn(out, outputs)
                    
            # Backpropagate errors computing gradients by accumulation
            loss.backward()

            # Use the gradients for one step of the optimizer
            self.optimizer.step()

            # Compute average loss. This keeps an average loss on an exponential window
            if (batch_idx == 0):
                loss_avg = loss.item()
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg

            # Update progress bar
            if (NVIDIA_SMI):
                usage = nvidia_smi.nvmlDeviceGetUtilizationRates(self.handle)
                memory = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
                t.set_postfix(loss=loss_avg, lr=current_lr, gpu=usage.gpu, memfree=f'{memory.free/1024**2:5.1f} MB', memused=f'{memory.used/1024**2:5.1f} MB')
            else:
                t.set_postfix(loss=loss_avg, lr=current_lr)
            
        return loss_avg

    def validate(self, epoch):
        """
        Validation

        Parameters
        ----------
        epoch : int
            Epoch
        
        """

        # Put the model in evaulation mode
        self.model_cuatro.eval()
        t = tqdm(self.validation_loader)
        n = 1
        loss_avg = 0.0

        # For evaluation we do not need to accumulate gradients
        with torch.no_grad():
            for batch_idx, (inputs, outputs) in enumerate(t):
                inputs = inputs.to(self.device)
                outputs = outputs.to(self.device)
                        
                out = self.model_cuatro(inputs)
            
                # Loss
                loss = self.loss_fn(out, outputs)

                if (batch_idx == 0):
                    loss_avg = loss.item()
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg
                
                t.set_postfix(loss=loss_avg)

        return loss_avg

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Train neural network')
    parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float,
                    metavar='LR', help='Learning rate')
    parser.add_argument('--wd', '--weigth-decay', default=0.0, type=float,
                    metavar='WD', help='Weigth decay')    
    parser.add_argument('--gpu', '--gpu', default=0, type=int,
                    metavar='GPU', help='GPU')
    parser.add_argument('--smooth', '--smoothing-factor', default=0.05, type=float,
                    metavar='SM', help='Smoothing factor for loss')
    parser.add_argument('--epochs', '--epochs', default=100, type=int,
                    metavar='EPOCHS', help='Number of epochs')
    parser.add_argument('--scheduler', '--scheduler', default=100, type=int,
                    metavar='SCHEDULER', help='Number of epochs before applying scheduler')
    parser.add_argument('--batch', '--batch', default=64, type=int,
                    metavar='BATCH', help='Batch size')
    
    parsed = vars(parser.parse_args())

    deepnet = Training(batch_size=parsed['batch'], gpu=parsed['gpu'], smooth=parsed['smooth'])

    deepnet.init_optimize(parsed['epochs'], lr=parsed['lr'], weight_decay=parsed['wd'], scheduler=parsed['scheduler'])
    deepnet.optimize()