#!/usr/bin/python3

import numpy as np
import glob
import os
import os.path
import sys
from sklearn.mixture import GaussianMixture as GM
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def draw_ellipse(position, covariance, ax=None, **kwargs):
    """Draw an ellipse with a given position and covariance"""
    ax = ax or plt.gca()
    
    # Convert covariance to principal axes
    if covariance.shape == (2, 2):
        U, s, Vt = np.linalg.svd(covariance)
        angle = np.degrees(np.arctan2(U[1, 0], U[0, 0]))
        width, height = 2 * np.sqrt(s)
    else:
        angle = 0
        width, height = 2 * np.sqrt(covariance)
    
    # Draw the Ellipse
    for nsig in range(1, 4):
        ax.add_patch(Ellipse(position, nsig * width, nsig * height,
                             angle, **kwargs))


def readQ(input_file):
    pos_tmp=np.genfromtxt(input_file,skip_header=9)
    q_Mg = [pos_tmp[i,6:] for i in range(len(pos_tmp)) if pos_tmp[i,1] == 1]
    q_Si = [pos_tmp[i,6:] for i in range(len(pos_tmp)) if pos_tmp[i,1] == 2]
    q_Ox = [pos_tmp[i,6:] for i in range(len(pos_tmp)) if pos_tmp[i,1] == 3]
    #q_tmp=pos_tmp[:,6:13]
    return q_Mg, q_Si, q_Ox

testStructure = []
dataPoints = [None]*3
dataPoints[0] = dict()
dataPoints[1] = dict()
dataPoints[2] = dict()
for base, dirs, files in os.walk('QSelectedReduced/'):
    for directories in dirs:
        if 'surf' in directories:
            continue
        list_file = sorted(glob.glob(base+directories+"/*Q.trj"))
        if len(list_file) == 0:
            continue
        dataPoints[0][directories] = np.empty((0, 20))
        dataPoints[1][directories] = np.empty((0, 20))
        dataPoints[2][directories] = np.empty((0, 20))
        for input_file in list_file:
            qmg, qsi, qox = readQ(input_file)
            dataPoints[0][directories] = np.append(dataPoints[0][directories], qmg, axis=0)
            dataPoints[1][directories] = np.append(dataPoints[1][directories], qsi, axis=0)
            dataPoints[2][directories] = np.append(dataPoints[2][directories], qox, axis=0)
        if 'LIQ' in list_file[0]:
            continue
        print(list_file[0])
        #testStructure.append(readQ(list_file[0]))

for key in dataPoints[0]:
    print(key)


for s in range(len(dataPoints)):
    fig, axs = plt.subplots(20, 20, figsize=(80, 80))
    fig.suptitle('Database plot [q2 ... q20]', fontsize=100)
    for i in range(20):
        for b in range(20):
            if b < i:
                continue
            for key in dataPoints[s]:
                axs[i, b].scatter(dataPoints[s][key][:,i], dataPoints[s][key][:,b], label=key)
            # for s in range(len(testStructure)):
            #     axs[i, b].scatter(testStructure[s][:,i], testStructure[s][:,b], c='black', marker='x')
            #axs[i, b].legend()
            if i+1 < 21:
                axs[i, b].set_xlabel('q'+str(i+1))
            else:
                axs[i, b].set_xlabel('q'+str(i+1-11)+' mono')
            if b+1 < 21:
                axs[i, b].set_ylabel('q'+str(b+1))
            else:
                axs[i, b].set_ylabel('q'+str(b+2-11)+' mono')
            print(i, b)
    fig.savefig('databaseQ'+str(s)+'.png', bbox_inches='tight')
