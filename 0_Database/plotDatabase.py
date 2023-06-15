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
    q_tmp=pos_tmp[:,5:]
    return q_tmp

testStructure = []
dataPoints = dict()
for base, dirs, files in os.walk('QValuesNonAverage/'):
    for directories in dirs:
        if 'surf' in directories:
            continue
        list_file = sorted(glob.glob(base+directories+"/*Q.trj"))
        if len(list_file) == 0:
            continue
        dataPoints[directories] = np.empty((0, 11))
        for input_file in list_file:
            dataPoints[directories] = np.append(dataPoints[directories], readQ(input_file), axis=0)
        if 'LIQ' in list_file[0]:
            continue
        print(list_file[0])
        testStructure.append(readQ(list_file[0]))

for key in dataPoints:
    print(key)


fig, axs = plt.subplots(11, 11, figsize=(60, 60))
fig.suptitle('Database plot [q2 ... q12] (Voronoi Cutoff)', fontsize=100)
for i in range(11):
    for b in range(11):
        if b < i:
            continue
        for key in dataPoints:
            axs[i, b].scatter(dataPoints[key][:,i], dataPoints[key][:,b], label=key)
        # for s in range(len(testStructure)):
        #     axs[i, b].scatter(testStructure[s][:,i], testStructure[s][:,b], c='black', marker='x')
        axs[i, b].legend()
        if i+2 < 13:
            axs[i, b].set_xlabel('q'+str(i+2))
        else:
            axs[i, b].set_xlabel('q'+str(i+2-11)+' mono')
        if b+2 < 13:
            axs[i, b].set_ylabel('q'+str(b+2))
        else:
            axs[i, b].set_ylabel('q'+str(b+2-11)+' mono')
        print(i, b)
fig.savefig('databaseQNonAverage.png', bbox_inches='tight')
