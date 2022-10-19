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

from models import *
from datasets import *
from ToftsModel import *

def main():
    with open("configs/training_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    cuda = torch.cuda.is_available()

    output_loc = "output/%s" % config['experiment_name']
    os.makedirs(output_loc, exist_ok = True)

    model = NeuralNetwork(input_dim = config['model']['input_dim'], output_dim = config['model']['output_dim'])
    if cuda:
        model.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    
    model.load_state_dict(torch.load("saved_models/%s/%d.pth" % (config['experiment_name'], config['testing']['epoch_test'])))

    criterion = nn.MSELoss()

    if config['dataset_name'] == "synthetic":
        dataloader = DataLoader(
            CustomDataset("datasplits/synthetic/test/curves.txt", "datasplits/synthetic/test/params.txt"),
            batch_size=1,
            shuffle=False,
            num_workers=config['training']['n_cpu'])

    total_loss = 0
    for i, batch in enumerate(dataloader):

        curves = Variable(batch["X"].type(Tensor))
        gt = Variable(batch["Y"].type(Tensor))

        output = model(curves)
        loss = criterion(output, gt)
        np.save(os.path.join(output_loc, str(i)), output.detach().cpu().numpy())
        total_loss += loss.item()
    
    print("Total loss for test set: %s" % total_loss)

    # Compare the output curve with a test curve
    '''
    current_dir = os.path.dirname(__file__)
    relative_path = "/output/test/1.npy"
    abs_file_path = os.path.join(current_dir, relative_path)
    '''

    predicted_params = np.load("output/test/49.npy")
    print(predicted_params.shape)
    AIF = np.load("AIF.npy")
    t = np.load("t.npy")
    predicted_curve = ToftsModel(predicted_params.T, t, AIF)
    test_curve = np.load("data/synthetic/curves/849.npy")
    plt.plot(t, test_curve, label="Test curve")
    plt.plot(t, predicted_curve, label="Predicted curve")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()