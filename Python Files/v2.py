
import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataloader import GetAllData
from sklearn.model_selection import train_test_split

#current working code for neural net

data = GetAllData()
X = data[0]
Y = data[1]
X2 = data[2]
Y2 = data[3]

'''
ax = plt.axes(projection='3d')
zdata = [j[0] for j in Y]
xdata = [j[0] for j in X]
ydata = [j[1] for j in X]
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
'''
#%%
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
y_train = torch.FloatTensor(y_train)
y_test = torch.FloatTensor(y_test)

'''
X_train = torch.FloatTensor(X)
X_test = torch.FloatTensor(X2)
y_train = torch.FloatTensor(Y)
y_test = torch.FloatTensor(Y2)
'''

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        self.fc1 = nn.Linear(3,16)
        self.fc2 = nn.Linear(16,64)
        self.fc3 = nn.Linear(64,16)
        self.fc4 = nn.Linear(16,1)
        
            
    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = nn.functional.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
model = MyModel()


criterion = nn.MSELoss()
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
#%%
#How well does the neural net function?
preds = []
with torch.no_grad():
   for val in X_test:
       y_hat = model.forward(val)
       preds.append([y_hat])
#Create test curve for IRL data comparison
preds2test = []
rltestrange = torch.FloatTensor([[3,7,j] for j in np.linspace(0,180,30)])
with torch.no_grad():
   for val in rltestrange:
       y_hat = model.forward(val)
       preds2test.append([y_hat])
print(criterion(torch.FloatTensor(np.linspace(0,180,30)),torch.flatten(torch.FloatTensor(Y2))))
#%%
#Plot some stuff
fig,(ax,ax2) = plt.subplots(2)
ax.plot(np.array(torch.flatten(y_test)),'.',label = 'Actual')
ax.plot([float(j[0]) for j in preds],'.',label = "Predicted")
ax.set_xlabel('Data Points')
ax.set_ylabel('Thrust (N)')
ax.legend(loc = 'upper right')
ax2.plot([float(j) for j in loss_arr])
ax2.set_xlabel('Epoch')
ax2.set_ylabel('MSE Loss')
criterion = nn.MSELoss()
print(criterion(torch.FloatTensor(preds),y_test))
fig.tight_layout()
fig.savefig('Essential Graphs.png')

#More plotting
fig2,ax3 = plt.subplots(1)
ax3.plot([dat[2] for dat in X2],Y2,'.',label = 'Experimental')
ax3.plot([3.951759099960327, 20.857494354248047, 23.939714431762695, 26.409557342529297, 28.817493438720703, 31.39550018310547, 34.209354400634766, 37.063377380371094, 39.9173583984375, 41.78074264526367, 43.611202239990234, 45.44166564941406, 47.27212905883789, 49.10258865356445, 50.921321868896484, 52.671775817871094, 54.42222213745117, 56.172672271728516, 57.719024658203125, 59.037742614746094, 60.50531005859375, 61.97745895385742, 63.44960403442383, 64.94293212890625, 66.44617462158203, 67.94942474365234, 69.45268249511719, 70.95592498779297, 72.45918273925781, 73.96244049072266],np.linspace(0,180,30),'.',label = 'Predicted')
ax3.legend()
ax3.set_xlabel("ESC Command")
ax3.set_ylabel("Thrust")
fig2.tight_layout()
#fig2.savefig("RLTest.png")