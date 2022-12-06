# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 09:52:21 2022

@author: Everett Werner
"""

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

#Intial neural net to take image of ggeometry as the input
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1,48,3)
        self.conv2 = nn.conv2d(48,96,3)
        self.conv3 = nn.conv2d(96,256)

        
        self.fc1 = nn.Linear(13,128)
        self.fc2 = nn.Linear(128,128)
        self.fc3 = nn.linear(128,1)
            
    def forward(self, image, data):
        x1 = self.cnn(image)
        x1 = nn.relu(self.conv1(x1)) #64*64*1 -> 64*64*48
        x1 = nn.max_pool2d(nn.relu(self.conv2(x1)),2) #64*64*96-> 32*32*496
        x1 = nn.max_pool2d(nn.relu(self.conv3(x1)),2) #32*32*256 -> 16*16*256
        x1 = nn.Linear(16*16*256,700)
        x1 = nn.Linear(700,200)
        x1 = nn.Linear(200,10)
        
        x2 = data
        
        x = torch.cat((x1, x2), dim=1)
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.relu(self.fc3(x))
        return x
    

#%%
training_loader = [[0,1],[1,2]]
model = MyModel().to(device)
loss_fn = torch.nn.MSELoss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = loss_fn(outputs, labels)
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000 # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    return last_loss