# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 15:07:20 2022

@author: Everett Werner
"""


import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataloader import GetAllData
from sklearn.model_selection import train_test_split

X = []
Y = []
for j in range(1,6):
    for k in range(0,70):
        X.append([float(j),float(k)])
        Y.append([np.exp(k/15)+2*k])

'''
ax = plt.axes(projection='3d')
zdata = [j[0] for j in Y]
xdata = [j[0] for j in X]
ydata = [j[1] for j in X]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
'''

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2,)
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.fc1 = nn.Linear(2,16)
        self.fc2 = nn.Linear(16,256)
        self.fc3 = nn.Linear(256,128)
        self.fc4 = nn.Linear(128,8)
        self.fc5 = nn.Linear(8,1)
        
            
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = nn.functional.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
model = MyModel()


criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=.01)

epochs = 5000
loss_arr = []
for i in range(epochs):
   y_hat = model.forward(X_train)
   loss = criterion(y_hat, y_train)
   loss_arr.append(loss)
 
   if i % 10 == 0:
       print(f'Epoch: {i} Loss: {loss}')
 
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()
   
preds = []
with torch.no_grad():
   for val in X_test:
       y_hat = model.forward(val)
       preds.append(y_hat)

fig,ax = plt.subplots(1)
ax.plot(np.array(torch.flatten(y_test)),'.',label = 'Actual')
ax.plot([float(j) for j in preds],'.')
ax.legend()