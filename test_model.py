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
    for i, batch in enumerate(dataloader):

        curves = Variable(batch["X"].type(Tensor))
        gt = Variable(batch["Y"].type(Tensor))

        output = model(curves)
        # print(output)
        loss = criterion(output, gt)
        network_loss[i] = loss

        if (loss > 1):
            # print(i)
            over_loss.append(i)

        elif (loss < 0.1):
            # print(i)
            under_loss.append(i)

        np.save(os.path.join(output_loc, str(i)), output.detach().cpu().numpy())

        # print(f"Loss value for value {i} is: {loss.item()}")
        total_loss += loss.item()
    
    print("Total loss for test set: %s" % total_loss)

    # Convert the loss lists to numpy arrays
    network_loss = network_loss.detach().cpu().numpy()
    print(network_loss)
    set_theme()
    plt.hist(network_loss, bins=100)
    plt.suptitle(f"Loss values per dataset for the test data")
    plt.title(f"Total loss: {round(total_loss, 2)}, Minimum loss: {round(np.min(network_loss), 4)}, Maximum loss: {round(np.max(network_loss), 2)}")
    plt.ylabel(f"Counts of overall sum: {len(dataloader)}")
    plt.xlabel("Loss function")
    plt.show()
    
    under_loss = np.array(under_loss)
    over_loss = np.array(over_loss)
    plot_loss = np.zeros((2, 3))
    plot_loss[0] = np.random.choice(under_loss, size = 3, replace = False)
    plot_loss[1] = np.random.choice(over_loss, size = 3, replace = False)

    fig, axis = plt.subplots(nrows = 2, ncols = 3, constrained_layout = True)

    for i in range(2):
        for j in range(3):

            # Compare the output curve with a test curve
            predicted_params = np.load("output/test/%d.npy" % (plot_loss[i][j]))
            print(predicted_params.shape)

            AIF = np.load("AIF.npy")
            t = np.load("t.npy")

            predicted_curve = ToftsModel(predicted_params.T, t, AIF)
            test_curve = np.load("data/synthetic/curves/%d.npy" % (plot_loss[i][j]))

            axis[i][j].plot(t, test_curve, label="Test curve")
            axis[i][j].plot(t, predicted_curve, label="Predicted curve")
            axis[i][j].set_title(f"Predicted vs Test curve at loss: {round(network_loss[int(plot_loss[i][j])], 3)}")
            axis[i][j].legend()

            # print(plot_loss[i][j])
            # print(round(network_loss[int(plot_loss[i][j])], 3))
            
    plt.show()

    # At the end, delete the .npy output files from the output directory 
    # This is done to avoid the overwriting of the files from the training
    # and testing classes
    print(output_loc)
    for file_name in os.listdir(output_loc):

        # Delete each file in the path corresponding to the 
        # output directory
        #print(file_name)
        file = output_loc + file_name
        if(os.path.isfile(file)):
            os.remove(file_name)
    # Now delete the empty directory
    os.rmdir(output_loc)

if __name__ == '__main__':
    main()