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

# Implement the early stopper class in order to avoid
# overfitting the network
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

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
        train_dataloader = DataLoader(
            CustomDataset("datasplits/synthetic/train/curves.txt", "datasplits/synthetic/train/params.txt"),
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['n_cpu'])

        validation_dataloader = DataLoader(
            CustomDataset("datasplits/synthetic/validate/curves.txt", "datasplits/synthetic/validate/params.txt"),
            shuffle=True,
            num_workers=config['training']['n_cpu'])
        
    prev_time = time.time()
    early_stopper = EarlyStopper(patience=3, min_delta=0.005)

    for epoch in range(config['training']['epoch'], config['training']['n_epochs']):
        for i, batch in enumerate(train_dataloader):

            curves = Variable(batch["X"].type(Tensor))
            gt = Variable(batch["Y"].type(Tensor))

            model.zero_grad()
            output = model(curves)
            loss = criterion(output, gt) 
            loss.backward()
            optimiser.step()
            print(f"\n Loss: {loss.item()} \n")

            # Determine approximate time left
            batches_done = epoch * len(train_dataloader) + i
            batches_left = config['training']['n_epochs'] * len(train_dataloader) - batches_done
            time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
            prev_time = time.time()
            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d] Loss: %f] ETA: %s"
                % (epoch, config['training']['n_epochs'], i, len(train_dataloader), loss.item(), time_left)
            )

        validation_loss = 0

        for i, batch in enumerate(validation_dataloader):

            curves = Variable(batch["X"].type(Tensor))
            gt = Variable(batch["Y"].type(Tensor))

            model.zero_grad()
            output = model(curves)
            loss = criterion(output, gt) 
            loss.backward()
            optimiser.step()  

            validation_loss += loss.item()

        validation_loss = validation_loss / len(validation_dataloader)          
        print(f"Validation loss at epoch {epoch} is: {validation_loss}")
        if early_stopper.early_stop(validation_loss):
            torch.save(model.state_dict(), "saved_models/%s/%d.pth" % (config['experiment_name'], epoch))             
            break

    torch.save(model.state_dict(), "saved_models/%s/%d.pth" % (config['experiment_name'], epoch))

if __name__ == '__main__':
    main()