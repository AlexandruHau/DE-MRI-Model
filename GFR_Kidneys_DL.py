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

def main():
    with open("configs/training_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    cuda = torch.cuda.is_available()
    if cuda: print ("GPU used") 

    os.makedirs("saved_models/%s" % config['experiment_name'], exist_ok = True)

    model = NeuralNetwork(input_dim = config['model']['input_dim'], output_dim = config['model']['output_dim'])
    if cuda:
        model.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

    if config['training']['epoch'] != 0:
        model.load_state_dict(torch.load("saved_models/%s/%d.pth" % (config['experiment_name'], config['epoch'])))

    optimiser = optim.SGD(model.parameters(), lr=config['training']['lr'])
    criterion = nn.MSELoss()

    if config['dataset_name'] == "synthetic":
        dataloader = DataLoader(
            CustomDataset("datasplits/synthetic/train/curves.txt", "datasplits/synthetic/train/params.txt"),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['n_cpu'])
        

    prev_time = time.time()
    for epoch in range(config['training']['epoch'], config['training']['n_epochs']):
        for i, batch in enumerate(dataloader):

            curves = Variable(batch["X"].type(Tensor))
            gt = Variable(batch["Y"].type(Tensor))

            model.zero_grad()
            output = model(curves)
            loss = criterion(output, gt)
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