# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 08:15:38 2022

@author: Everett Werner
"""
import os as os

import numpy as np
import matplotlib.pyplot as plt

def GetAllData(plot=False):
    basepath = r"C:\Users\Everett Werner\Desktop\Current School\ML\Data"
    folders = [name for name in os.listdir(basepath)]
    print(folders)

    slot1dict = {}
    slot2dict = {}
    slot3dict = {}
    slot4dict = {}
    slot5dict = {}
    slot6dict = {}
    slot7dict = {}


    
    bigdict = [slot1dict,slot2dict,slot3dict,slot4dict,slot5dict,slot6dict,slot7dict]
    for foldernum,folder in enumerate(folders): #iterate through storage folders
        cfolder  = os.path.join(basepath,folder)
        for filenum,filename in enumerate(os.listdir(cfolder)): #iterate through files in folder
            f = os.path.join(cfolder, filename)
            with open(f) as file:
                lines = file.readlines()
                #cut list into list of lists at 'power'
                size = len(lines)
                idx_list = [idx + 1 for idx, val in
                            enumerate(lines) if val == 'power\n']
                res = [lines[i: j] for i, j in
                       zip([0] + idx_list, idx_list + 
                       ([size] if idx_list[-1] != size else []))]
                nomsize = len(res[10])
                res = [dat for dat in res if len(dat)==nomsize]
                cleandat = []
                for powers in res: #take average of on and off data and subtract
                    avg1 = []
                    avg2 = []
                    check = 0
                    powers = [j.strip() for j in powers]
                    for j in powers:
                        if j == 'clear':
                            check = 1
                        if check != 1:
                            avg1.append(float(j))
                        if check == 1 and j != 'power' and j != '' and j != 'clear':
                            avg2.append(float(j))
                    avg1 = np.average(avg1[1:])
                    avg2 = np.average(avg2[1:])
                    avg = avg1-avg2
                    cleandat.append([float(powers[0]),avg])
                bigdict[foldernum][filename] = cleandat
    foldernames = [slot1dict,slot2dict,slot3dict,slot4dict,slot5dict,slot6dict,slot7dict]
    
    if plot:
        fig,ax = plt.subplots(1)
        ax.plot([j[0] for j in plot],[j[1] for j in plot],'.')
    
    inputs = []
    outputs = []
    
    inputs2 = []
    outputs2 = []
    for cf,f in enumerate(foldernames):
        cslot = cf+1
        for cdat in f.keys():
            if 'none' in cdat:
                attach = 0
            elif 'arm' in cdat:
                attach = 1
            elif 'disk' in cdat:
                attach = 2
            elif 'tri' in cdat:
                attach = 3
            elif 'diks' in cdat:
                attach = 4
            elif 'long' in cdat:
                attach = 5
            else:
                print('Unknown string: %' % cdat)
            if cslot != 6:
                for thrusts in f[cdat]:
                    inputs.append([float(attach),float(cslot),float(thrusts[1])])
                    outputs.append([thrusts[0]])
            if cslot == 6:
                for thrusts in f[cdat]:
                    inputs2.append([float(3),float(7),float(thrusts[0])])
                    outputs2.append([thrusts[1]])
    return (inputs,outputs,inputs2,outputs2)
    
                
                    