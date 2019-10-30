# This code is provided for Deep Learning class (CS 482/682) Homework 6 practice.
# This is a sketch code for main function. There are some given hyper-parameters insideself.
# You need to finish the design and train your network.
# @Copyright Cong Gao, the Johns Hopkins University, cgao11@jhu.edu
# Modified by Hongtao Wu on Oct 11, 2019 for Fall 2019 Machine Learning: Deep Learning HW6

import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


######################## Hyperparameters #################################
# Batch size can be changed if it does not match your memory, please state your batch step_size
# in your report.
train_batch_size = 10
validation_batch_size=10
# Please use this learning rate for Prob1(a) and Prob1(b)
learning_rate = 0.001
# This num_epochs is designed for running to be long enough, you need to manually stop or design
# your early stopping method.
num_epochs = 1000

# TODO: Design your own dataset
class ImageDataset(Dataset):
    def __init__(self, *):

    def __len__ (self):
        return *

    def __getitem__(self, idx):
        return *


# TODO: Implement DICE loss
class DICELoss(nn.Module):
    def __init__(self, *):
    
    def forward(self, input, target):
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()

        return 1 - ((2. * intersection + smooth) /
                    (iflat.sum() + tflat.sum() + smooth))


# TODO: Use your designed dataset for dataloading
train_dataset=ImageDataset(input_dir = ***)
validation_dataset=ImageDataset(input_dir = *** )

optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


print("Start Training...")
for epoch in range(num_epochs):
    ########################### Training #####################################
    print("\nEPOCH " +str(epoch+1)+" of "+str(num_epochs)+"\n")
    # TODO: Design your own training section


    ########################### Validation #####################################
    # TODO: Design your own validation section
