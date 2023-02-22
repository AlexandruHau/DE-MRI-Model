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
# from ToftsModel import *

def main():

    # Open the config object used for parsing through the 
    # .yaml file to extract the network parameters
    with open("configs/training_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    print(config)

    # Check whether the cuda library is available or not
    cuda = torch.cuda.is_available()

    # Use the concatenated dataloader to split it into 5 parts - out
    # of these parts, 4 splits are used for training and one for testing
    splits = KFold(n_splits = 4, shuffle = True, random_state = 42)

    # Work with each split from the cross validation
    for fold in range(4):

        # Create numpy arrays for storing the test parameters, the
        # predicted parameters, as well as the loss function for each test
        input_params = np.zeros((2500, 4))
        predicted_params = np.zeros((2500, 4))
        loss_values = np.zeros(2500)

        # Create a neural network model of input and output 
        # dimensions as specified in the .yaml file
        model = NeuralNetwork(input_dim = config['model']['input_dim'], output_dim = config['model']['output_dim'])
        if cuda:
            model.cuda()

        Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

        # Set the optimiser method by specifying the parameters from 
        # the .yaml file (particularly the learning rate) Also need to
        # set the criterion function as the Mean Squared Error MSE()
        criterion = nn.MSELoss()
        optimiser = optim.SGD(model.parameters(), lr=config['training']['lr'])

        # Create a separate pytorch dataloader for the testing dataset
        # Use the DataLoader class from torch. utils.data as well as the 
        # .txt files for the curves and set of parameters specific to the curves
        test_loader = DataLoader(
                CustomDataset("datasplits/synthetic/fold%d/test/curves.txt" % fold, "datasplits/synthetic/fold%d/test/params.txt" % fold),
                shuffle=True,
                num_workers=config['training']['n_cpu'])

        # Create a new directory called "output" with the string
        # followed by the name of the experiment whose value is 
        # taken from the .yaml file
        output_loc = "output_fold/%d/%s" % (fold, config['experiment_name'])
        os.makedirs(output_loc, exist_ok = True)

        # Load now the content of the neural network from the 
        # .pth file - as saved before hand in the main file. Also 
        # set the criterion function as the Mean Squared Error MSE()
        model.load_state_dict(torch.load("saved_models/%s/%d.pth" % (config['experiment_name'], config['testing']['epoch_test'])))
        print(f"Fold: {fold}")
        
        for i, batch in enumerate(test_loader):

            curves = Variable(batch["X"].type(Tensor))
            gt = Variable(batch["Y"].type(Tensor))

            output = model(curves)
            loss = criterion(output, gt)

            # Now store the input and predicted values together with the
            # loss values for the training datasets
            # int(0.25 * len(input_params) * fold + i)
            input_params[i] = gt.detach().cpu().numpy()
            predicted_params[i] = output.detach().cpu().numpy()
            loss_values[i] = loss.detach().cpu().numpy()

            print(f"Test loss {loss_values[i]} at position: {i}")
            print(f"Input params: {input_params[i]}")
            print(f"Predicted params: {predicted_params[i]}")

        # Now save all the required parameters
        np.savetxt('CrossValidationEvaluation_Params_%d_Input.txt' % fold, input_params, delimiter=',', fmt='%f')
        np.savetxt('CrossValidationEvaluation_Params_%d_Predict.txt' % fold, predicted_params, delimiter=',', fmt='%f')

    '''
    df_curves = pd.DataFrame({"Input curves" : input_params.flatten(), "Predicted curves" : predicted_params.flatten()})
    df_losses = pd.DataFrame({"Losses" : loss_values.flatten()})
    df_curves.to_csv("CrossValidationEvaluation_Params.csv")
    df_losses.to_csv("CrossValidationEvaluation_Losses.csv")
    '''

if __name__ == '__main__':
    main()