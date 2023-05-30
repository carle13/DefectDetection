#!/usr/bin/python3

from joblib import Parallel, delayed
import numpy as np
import glob
import os
import os.path
import sys
from sklearn.mixture import GaussianMixture as GM

#python3 ComputeProbability.py 2_Relaxation/
inDir = os.path.join(sys.argv[1], '')

def readQ(input_file):
    pos_tmp=np.genfromtxt(input_file,skip_header=9)
    q_tmp=pos_tmp[:,6:]
    return q_tmp

categories = []
testStructure = []
numDirs = 0
X = np.empty((0,14))
for base, dirs, files in os.walk('0_Database/QValues/'):
    numDirs += len(dirs)
    for directories in dirs:
        list_file = sorted(glob.glob(base+directories+"/*Q.trj"))
        if len(list_file) == 0:
            numDirs -= 1
            continue
        for input_file in list_file:
            X = np.append(X, readQ(input_file), axis=0)
        testStructure.append(readQ(list_file[0]))
        categories.append(directories)
    
print('Number of clusters used in the model: ', numDirs)


#Creating and training the GM model
dirModel = '0_GMModel/Voronoi/'
model = GM(numDirs, n_init=100)
if not os.path.exists('0_GMModel/Voronoi/weights.npy'):
    model.fit(X)
else:
    means = np.load(dirModel+'means.npy')
    covar = np.load(dirModel+'covariances.npy')
    model.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covar))
    model.weights_ = np.load(dirModel+'weights.npy')
    model.means_ = means
    model.covariances_ = covar

#Test and identify clusters
clusterIDs = ['']*len(categories)
for i in range(len(categories)):
    testProbs = model.predict_proba(testStructure[i])
    aveProb = np.mean(testProbs, axis=0)
    print('Test structure: '+categories[i])
    print('Probabilities: ', aveProb)
    print('Max Prob: '+str(np.max(aveProb))+'   Cluster: '+str(np.argmax(aveProb)))
    clusterIDs[np.argmax(aveProb)] = categories[i]

print('Identified clusters: ', clusterIDs)
for i in range(len(clusterIDs)):
    if clusterIDs[i] == '':
        print('Not all clusters have been recognized!')
        exit()
print('Identified clusters: ', clusterIDs)

#Save successfully trained model
if not os.path.exists('0_GMModel/Voronoi/weights.npy'):
    os.makedirs('0_GMModel/Voronoi/', exist_ok=True)
    np.save(dirModel+'weights.npy', model.weights_, allow_pickle=False)
    np.save(dirModel+'means.npy', model.means_, allow_pickle=False)
    np.save(dirModel+'covariances.npy', model.covariances_, allow_pickle=False)


# Compute probabilities for each atom and write to output file
def computeProbabilityPerAtom(input_file):
    # Define output file name
    output_file = input_file.replace(".Q", ".PROB")
    output_file = output_file.replace('QValues', 'Output')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(output_file)
    # Read q values from input files
    pos=np.genfromtxt(input_file,skip_header=9)
    q=pos[:,6:]
    # Predict probabilities for all atoms in the system
    try:
        probs = model.predict_proba(q)
    except:
        return

    # Start writing to output file
    os.system("head -n 8 "+input_file+"  > "+output_file)
    Pstring = ' '.join(['p'+clusterIDs[i] for i in range(len(clusterIDs))])
    os.system("echo  'ITEM: ATOMS id type x y z coord "+Pstring+"' >> "+output_file)
    with open(output_file,'a') as fw:
        for i in range(len(probs)):
            # Write results in file
            PatomString = ' '.join([str(probs[i][b]) for b in range(len(probs[i]))])
            fw.write("%g %g %g %g %g %g "%(pos[i,0],pos[i,1],pos[i,2],pos[i,3],pos[i,4], pos[i,5])+PatomString+"\n")
    # print(input_file,np.mean(D_WRZ),np.mean(D_BCT),np.mean(D_MC),np.mean(D_SC))

#Iterate through all files in Q directory
list_file = sorted(glob.glob(inDir+'*/QValues/**/*.Q.trj', recursive=True))
for input_file in list_file:
    computeProbabilityPerAtom(input_file)

#Iterate through all files in Q directory
list_file = sorted(glob.glob(inDir+'QValues/**/*.Q.trj', recursive=True))
for input_file in list_file:
    computeProbabilityPerAtom(input_file)