# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 08:15:38 2022

@author: Everett Werner
"""
#Import and loads data for offset visualization
import os as os
import numpy as np
import matplotlib.pyplot as plt

xdat = []
ydat = []
basepath = r"C:\Users\Everett Werner\Desktop\Current School\ML\Data"
folders = [name for name in os.listdir(basepath)]
with open(basepath+'\\0force.txt.log') as file:
    lines = file.readlines()
    odd = 1
    for j in lines:
        if odd == 1:
            ydat.append(float(j)/100)
            odd = 0
        else:
            xdat.append(float(j)/1000)
            odd =1
fig,ax = plt.subplots(1)
ax.plot(xdat,ydat)
ax.set_xlabel("Time (s)")
ax.set_ylabel("Reading (N)")
ax.grid()
fig.tight_layout()
fig.savefig('0loaddat.png')
    
