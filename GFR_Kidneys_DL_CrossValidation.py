import os
import sys
import time
import datetime

import yaml

import torch
from torch import nn
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import KFold

from models import *
from datasets import *

def main():
    with open("configs/training_config.yaml", "r") as stream:
        config = yaml.safe_load(stream)

    cuda = torch.cuda.is_available()
    if cuda: print ("GPU used") 

    # Set up a fixed seed for generating the random selection
    # of elements in the k folds / splits
    torch.manual_seed(42)

    # Create the directory for the saved network model - or don t create it if
    # it already exists
    os.makedirs("saved_models/%s" % config['experiment_name'], exist_ok = True)

    # Import the parameters from the .yaml file and the class from models.py file
    # to design the architecture of the network
    model = NeuralNetwork(input_dim = config['model']['input_dim'], output_dim = config['model']['output_dim'])
    if cuda:
        model.cuda()

    batch_size = config['training']['batch_size']

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

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

    # Use the concatenated dataloader to split it into 5 parts - out
    # of these parts, 4 splits are used for training and one for testing
    splits = KFold(n_splits = 5, shuffle = True, random_state = 42)

    # Work with each split from the cross validation
    print(splits.split(np.arange(len(dataloader))))
    for fold, (train_idx, val_idx) in enumerate(splits.split(range(1, 1000))):

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(val_idx)
        train_loader = DataLoader(dataloader.dataset, batch_size=batch_size, sampler=train_sampler)
        test_loader = DataLoader(dataloader.dataset, batch_size=batch_size, sampler=test_sampler)

        print(f"Fold: {fold}")
        print(val_idx[0])
        print(train_idx)
        
        prev_time = time.time()
        for epoch in range(config['training']['epoch'], config['training']['n_epochs']):
            for i, batch in enumerate(train_loader):

                curves = Variable(batch["X"].type(Tensor))
                gt = Variable(batch["Y"].type(Tensor))

                model.zero_grad()
                output = model(curves)
                loss = criterion(output, gt)
                loss.backward()
                optimiser.step()

                # Determine approximate time left
                batches_done = epoch * len(train_loader) + i
                batches_left = config['training']['n_epochs'] * len(train_loader) - batches_done
                time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
                prev_time = time.time()
                
                # Print log
                # sys.stdout.write(
                #     "\r[Epoch %d/%d] [Batch %d/%d] Loss: %f] ETA: %s"
                #     % (epoch, config['training']['n_epochs'], i, len(train_loader), loss.item(), time_left)
                # )

        torch.save(model.state_dict(), "saved_models/%s/%d_%d.pth" % (config['experiment_name'], epoch, fold))

if __name__ == '__main__':
    main()