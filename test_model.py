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

    # Create a new directory called "output" with the string
    # followed by the name of the experiment whose value is 
    # taken from the .yaml file
    output_loc = "output/%s" % config['experiment_name']
    os.makedirs(output_loc, exist_ok = True)

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

    # Use the DataLoader class from the pytorch library in order to
    # load the .txt files for the curves and training paths. For each path,
    # extract the curves and the parameters
    if config['dataset_name'] == "synthetic":
        dataloader = DataLoader(
            CustomDataset("datasplits/synthetic/test/curves.txt", "datasplits/synthetic/test/params.txt"),
            batch_size=1,
            shuffle=False,
            num_workers=config['training']['n_cpu'])

    total_loss = 0

    # Initialize arrays for loss over 1 (predicted model goes wrong)
    # and for loss less than 0.1 (predicted model good enough)
    over_loss = []
    under_loss = []
    
    # Initialize an empty tensor
    cuda0 = torch.device('cuda:0')
    network_loss = torch.zeros(len(dataloader), dtype = torch.float64, device = cuda0)
    count = 0
    for i, batch in enumerate(dataloader):

        curves = Variable(batch["X"].type(Tensor))
        gt = Variable(batch["Y"].type(Tensor))
        output = model(curves)
        loss = criterion(output, gt)
        network_loss[i] = loss

        np.save(os.path.join(output_loc, str(i)), output.detach().cpu().numpy())

        print(f"Loss value for value {i} is: {loss.item()}")
        total_loss += loss.item()

        print(count)
        count += 1
    
    print("Total loss for test set: %s" % total_loss)

    AIF = np.load("AIF.npy")
    t = np.load("t.npy")
    tnew = np.load("tnew.npy")

    predicted_params = np.zeros((3000, 4))
    input_params = np.zeros((3000, 4))

    for i, batch in enumerate(dataloader):

        # Compare the output curve with a test curve
        predicted_params[i] = np.load("output/test/%d.npy" % (i))
        input_params[i] = np.load("data/synthetic/params/%d.npy" % (i + 7000))
    
    df = pd.DataFrame({"Input params" : input_params.flatten(), "Predicted params" : predicted_params.flatten()})
    df.to_csv("Parameters_GFR.csv")

if __name__ == '__main__':
    main()