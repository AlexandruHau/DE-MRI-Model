## Implementation of paper: https://ieeexplore.ieee.org/document/8563412/
## Adam Tattersall - 12/09/2022
## Req. packages
    # pyyaml
    # numpy
    # pytorch

import os
import time

import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from seaborn import set_theme
import random
import pandas as pd

from models import *
from datasets import *
from ToftsModel import *

def main():

    # Open the config object used for parsing through the 
    # .yaml file to extract the network parameters
    with open("configs/training_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    print(config)

    # Check whether the cuda library is available or not
    cuda = torch.cuda.is_available()

    # Create a neural network model of input and output 
    # dimensions as specified in the .yaml file
    model = NeuralNetwork(input_dim = config['model']['input_dim'], output_dim = config['model']['output_dim'])
    if cuda:
        model.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    # Load now the content of the neural network from the 
    # .pth file - as saved before hand in the main file. Also 
    # set the criterion function as the Mean Squared Error MSE()
    model.load_state_dict(torch.load("saved_models/%s/%d.pth" % (config['experiment_name'], config['testing']['epoch_test'])))

    criterion = nn.MSELoss()
    # batch_size = config['training']['batch_size']

    # Create two separate pytorch dataloaders - one for the training dataset
    # and one for the test dataset. Use the DataLoader class from torch.
    # utils.data as weel as the .txt files for the curves and set of parameters
    # specific to the curves
    dataloader = DataLoader(
            CustomDataset("datasplits/data/curves.txt", "datasplits/data/params.txt"),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['n_cpu'])   

    optimiser = optim.SGD(model.parameters(), lr=config['training']['lr'])
    criterion = nn.MSELoss() 

    # Set up the required arrays for saving the predicted curves and
    # parameters together with the ground truth (input)
    ground_truth_curves = np.zeros((1000, 150))
    ground_truth_params = np.zeros((1000, 4))
    predicted_curves = np.zeros((1000, 150))
    predicted_params = np.zeros((1000, 4))

    # Use the concatenated dataloader to split it into 5 parts - out
    # of these parts, 4 splits are used for training and one for testing
    splits = KFold(n_splits = 5, shuffle = True, random_state = 42)

    # Load the time and AIF numpy arrays
    t = np.load("t.npy")
    AIF = np.load("AIF.npy")

    # Work with each split from the cross validation
    print(splits.split(np.arange(len(dataloader))))
    for fold, (train_idx, val_idx) in enumerate(splits.split(range(1, 1000))):

        # Create a new directory called "output" with the string
        # followed by the name of the experiment whose value is 
        # taken from the .yaml file
        output_loc = "output_fold/%d/%s" % (fold, config['experiment_name'])
        os.makedirs(output_loc, exist_ok = True)

        # Load now the content of the neural network from the 
        # .pth file - as saved before hand in the main file. Also 
        # set the criterion function as the Mean Squared Error MSE()
        model.load_state_dict(torch.load("saved_models/%s/%d.pth" % (config['experiment_name'], config['testing']['epoch_test'])))

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataloader.dataset, sampler=train_sampler)
        test_loader = DataLoader(dataloader.dataset, sampler=test_sampler)

        print(f"Fold: {fold}")
        
        for i, batch in enumerate(test_loader):

            # print(i)
            curves = Variable(batch["X"].type(Tensor))
            gt = Variable(batch["Y"].type(Tensor))

            output = model(curves)
            loss = criterion(output, gt)
            
            ground_truth_curves[fold * len(test_loader) + i] = np.load("data/synthetic/curves/%d.npy" % val_idx[i])
            ground_truth_params[fold * len(test_loader) + i] = np.load("data/synthetic/params/%d.npy" % val_idx[i])
            predicted_params[fold * len(test_loader) + i] = output.detach().cpu().numpy()
            predicted_params[fold * len(test_loader) + i].T[0] = predicted_params[fold * len(test_loader) + i].T[0] / (K_calibration)
            predicted_curves[fold * len(test_loader) + i] = ToftsModel(predicted_params[fold * len(test_loader) + i], t, AIF)

    # Save all the numpy array using the pandas library - save them in .csv format
    df_curves = pd.DataFrame({"Input curves" : ground_truth_curves.flatten(), "Predicted curves" : predicted_curves.flatten()})
    df_params = pd.DataFrame({"Input params" : ground_truth_params.flatten(), "Predicted curves" : predicted_params.flatten()})

    df_curves.to_csv("CrossValidationEvaluation_Curves.csv")
    df_params.to_csv("CrossValidationEvaluation_Params.csv")

if __name__ == '__main__':
    main()