## Implementation of paper: https://ieeexplore.ieee.org/document/8563412/
## Adam Tattersall - 12/09/2022
## Req. packages
    # pyyaml
    # numpy
    # pytorch

import os
import sys
import time
import datetime

import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable

from models import *
from datasets import *
from ToftsModel import *

def main():

    # Open the AIF and temporal files
    AIF = np.load("AIF.npy")
    t = np.load("t.npy")

    # Open the .yaml file and extract the parameters
    with open("configs/training_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    # Check if the cuda library is available - if yes, run
    # the pytorch network on GPU
    cuda = torch.cuda.is_available()
    if cuda: print ("GPU used") 

    # Make a directory where the model for the experiment (epoch)
    # is saved. Afterwards, create the model using the parameters
    # from the .yaml file with the input and output dimensions
    os.makedirs("saved_models/%s" % config['experiment_name'], exist_ok = True)

    model = NeuralNetwork(input_dim = config['model']['input_dim'], output_dim = config['model']['output_dim'])
    if cuda:
        model.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    if config['training']['epoch'] != 0:
        model.load_state_dict(torch.load("saved_models/%s/%d.pth" % (config['experiment_name'], config['epoch'])))

    # Declare the optimizer used for the neural network, with the 
    # learning rate specified from the .yaml file. As loss function, the
    # mean squared error is implemented
    optimiser = optim.SGD(model.parameters(), lr=config['training']['lr'])
    criterion = nn.MSELoss()

    # Implement the dataloader method from pytorch to load the input data as
    # the .npy files describing the curves and the parameters. Also declare the
    # batch size and the number of cpus available
    if config['dataset_name'] == "synthetic":
        dataloader = DataLoader(
            CustomDataset("datasplits/synthetic/train/curves.txt", "datasplits/synthetic/train/params.txt"),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['n_cpu'])
        
    # Now start iterating through the epochs - have 200 epochs required for 
    # training the neural network
    prev_time = time.time()
    for epoch in range(config['training']['epoch'], config['training']['n_epochs']):

        # For each epoch, have the batch size of 50 as specified in the
        # .yaml file and the set of 20 batches -> For each batch, have the 
        # X array of [50, 150] corresponding to the convolution between the
        # AIF and the VIRF, and the Y array of [50, 4] corresponding to the
        # parameters for the curve
        for i, batch in enumerate(dataloader):

            curves = Variable(batch["X"].type(Tensor))
            gt = Variable(batch["Y"].type(Tensor))
            
            # Concatenate the parameter and the curves tensors from the
            # training input into one tensor
            test_tensor = torch.cat([gt, curves], dim = 1)
            model.zero_grad()
            output = model(curves)

            # Make a copy of the output tensor of size (50, 154) - 4 elements
            # representing the parameters of the curve and 150 elements representing
            # the curve array / linear tensor. NOTE: 50 represents the size of the batch.
            # Then, the last 150 elements are updated using the predicted curve parameters
            # (1st four elements of the output)
            np_output = output.clone().detach().cpu().numpy()
            for i in range(np_output.shape[0]):
                np_output[i][4:] = ToftsModel(np_output[i][:4], t, AIF)
            output[4:] = torch.from_numpy(np_output[4:]).to(output)

            # The loss of tensor merging both the predicted parameters and predicted 
            # curve is worked out
            loss = criterion(output, test_tensor)
            loss.backward()
            optimiser.step()

            # Determine approximate time left
            batches_done = epoch * len(dataloader) + i
            batches_left = config['training']['n_epochs'] * len(dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] Loss: %f] ETA: %s"
                % (epoch, config['training']['n_epochs'], i, len(dataloader), loss.item(), time_left)
            )

    torch.save(model.state_dict(), "saved_models/%s/%d.pth" % (config['experiment_name'], epoch))

if __name__ == '__main__':
    main()
