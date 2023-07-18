#!/usr/bin/python3

from joblib import Parallel, delayed
import numpy as np
import glob
import os
import os.path
import sys
from sklearn.mixture import GaussianMixture as GM
from sklearn.model_selection import GridSearchCV

#python3 ComputeProbability.py 2_Relaxation/
inDir = os.path.join(sys.argv[1], '')


#Save training information to file
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

if not os.path.exists('0_GMModel/Cutoff/training.out'):
    os.makedirs('0_GMModel/Cutoff/', exist_ok=True)
    f = open('0_GMModel/Cutoff/training.out', 'w')
    sys.stdout = Tee(sys.stdout, f)

#   1 ---- Mg
#   2 ---- Si
#   3 ---- O


def readQ(input_file, startCol=6):
    pos_tmp=np.genfromtxt(input_file,skip_header=9)
    #return three lists for Mg, Si, and O
    q_Mg = [pos_tmp[i,startCol:] for i in range(len(pos_tmp)) if pos_tmp[i,1] == 1]
    q_Si = [pos_tmp[i,startCol:] for i in range(len(pos_tmp)) if pos_tmp[i,1] == 2]
    q_Ox = [pos_tmp[i,startCol:] for i in range(len(pos_tmp)) if pos_tmp[i,1] == 3]
    #q_tmp=pos_tmp[:,6:13]
    return q_Mg, q_Si, q_Ox


def gmm_bic_score(estimator, X):
    return -estimator.aic(X)

param_grid = {"n_components": range(4, 10),}


categories = []
testStructure = []
numDirs = 0
MgStructs = []
SiStructs = []
OxStructs = []
XMg = np.empty((0,20))
XSi = np.empty((0,20))
XOx = np.empty((0,20))
for base, dirs, files in os.walk('0_Database/QSelectedReduced/'):
    numDirs += len(dirs)
    for directories in dirs:
        list_file = sorted(glob.glob(base+directories+"/*Q.trj"))
        if len(list_file) == 0:
            numDirs -= 1
            continue
        MgStructs.append(np.empty((0,20)))
        SiStructs.append(np.empty((0,20)))
        OxStructs.append(np.empty((0,20)))
        for input_file in list_file:
            qmg, qsi, qox = readQ(input_file)
            XMg = np.append(XMg, qmg, axis=0)
            XSi = np.append(XSi, qsi, axis=0)
            XOx = np.append(XOx, qox, axis=0)
            MgStructs[-1] = np.append(MgStructs[-1], qmg, axis=0)
            SiStructs[-1] = np.append(SiStructs[-1], qsi, axis=0)
            OxStructs[-1] = np.append(OxStructs[-1], qox, axis=0)
            #if 'Dislo' in input_file:
            #print('including more times the dislocations')
            for repeat in range(4):
                XMg = np.append(XMg, qmg, axis=0)
                XSi = np.append(XSi, qsi, axis=0)
                XOx = np.append(XOx, qox, axis=0)
                MgStructs[-1] = np.append(MgStructs[-1], qmg, axis=0)
                SiStructs[-1] = np.append(SiStructs[-1], qsi, axis=0)
                OxStructs[-1] = np.append(OxStructs[-1], qox, axis=0)
        print('Number of atoms in each structure')
        print(directories)
        print('Mg: ', MgStructs[-1].shape[0])
        print('Si: ', SiStructs[-1].shape[0])
        print('Ox: ', OxStructs[-1].shape[0])
        print('------------------------------------')
        #testStructure.append(readQ(list_file[0]))
        categories.append(directories)


#Find out number of clusters
grid_seach = GridSearchCV(GM(), param_grid=param_grid, scoring=gmm_bic_score)
grid_seach.fit(XMg)
numClustersMg = grid_seach.best_params_['n_components']
grid_seach = GridSearchCV(GM(), param_grid=param_grid, scoring=gmm_bic_score)
grid_seach.fit(XSi)
numClustersSi = grid_seach.best_params_['n_components']
grid_seach = GridSearchCV(GM(), param_grid=param_grid, scoring=gmm_bic_score)
grid_seach.fit(XOx)
numClustersOx = grid_seach.best_params_['n_components']
print('Number of clusters used in each model: ')
print('Mg --- ', numClustersMg)
print('Si --- ', numClustersSi)
print('Ox --- ', numClustersOx)
print('--------------------------------')

print('Datapoints Mg: ', XMg.shape)
print('Datapoints Si: ', XSi.shape)
print('Datapoints Ox: ', XOx.shape)
# print(len(testStructure[0]))

#Creating and training the GM model
dirModel = '0_GMModel/Cutoff/'
modelMg = GM(numClustersMg, n_init=5)
modelSi = GM(numClustersSi, n_init=5)
modelOx = GM(numClustersOx, n_init=5)

if not os.path.exists('0_GMModel/Cutoff/weightsMg.npy'):
    modelMg.fit(XMg)
    modelSi.fit(XSi)
    modelOx.fit(XOx)
else:
    meansMg = np.load(dirModel+'meansMg.npy')
    meansSi = np.load(dirModel+'meansSi.npy')
    meansOx = np.load(dirModel+'meansOx.npy')
    covarMg = np.load(dirModel+'covariancesMg.npy')
    covarSi = np.load(dirModel+'covariancesSi.npy')
    covarOx = np.load(dirModel+'covariancesOx.npy')
    modelMg.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covarMg))
    modelMg.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covarSi))
    modelMg.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(covarOx))
    modelMg.weights_ = np.load(dirModel+'weightsMg.npy')
    modelSi.weights_ = np.load(dirModel+'weightsSi.npy')
    modelOx.weights_ = np.load(dirModel+'weightsOx.npy')
    modelMg.means_ = meansMg
    modelSi.means_ = meansSi
    modelOx.means_ = meansOx
    modelMg.covariances_ = covarMg
    modelSi.covariances_ = covarSi
    modelOx.covariances_ = covarOx

#Test and identify clusters
clusterIDsMg = ['']*numClustersMg
clusterIDsSi = ['']*numClustersSi
clusterIDsOx = ['']*numClustersOx
prevClustMg = np.zeros(numClustersMg)
prevClustSi = np.zeros(numClustersSi)
prevClustOx = np.zeros(numClustersOx)
for i in range(len(MgStructs)):
    testProbsMg = modelMg.predict_proba(MgStructs[i])
    testProbsSi = modelSi.predict_proba(SiStructs[i])
    testProbsOx = modelOx.predict_proba(OxStructs[i])
    aveProbMg = np.mean(testProbsMg, axis=0)
    aveProbSi = np.mean(testProbsSi, axis=0)
    aveProbOx = np.mean(testProbsOx, axis=0)
    print('-----------------------')
    print('Test structure: '+categories[i])
    print('Probabilities Mg: ', aveProbMg)
    for b in range(len(clusterIDsMg)):
        if aveProbMg[b] > prevClustMg[b]:
            clusterIDsMg[b] = categories[i]
            prevClustMg[b] = aveProbMg[b]
    #print('Max Prob Mg: '+str(np.max(aveProbMg))+'   Cluster: '+str(np.argmax(aveProbMg)))
    # clusterIDsMg[np.argmax(aveProbMg)] = categories[i]
    print('Probabilities Si: ', aveProbSi)
    for b in range(len(clusterIDsSi)):
        if aveProbSi[b] > prevClustSi[b]:
            clusterIDsSi[b] = categories[i]
            prevClustSi[b] = aveProbSi[b]
    #print('Max Prob Si: '+str(np.max(aveProbSi))+'   Cluster: '+str(np.argmax(aveProbSi)))
    # clusterIDsSi[np.argmax(aveProbSi)] = categories[i]
    print('Probabilities Ox: ', aveProbOx)
    for b in range(len(clusterIDsOx)):
        if aveProbOx[b] > prevClustOx[b]:
            clusterIDsOx[b] = categories[i]
            prevClustOx[b] = aveProbOx[b]
    #print('Max Prob Ox: '+str(np.max(aveProbOx))+'   Cluster: '+str(np.argmax(aveProbOx)))
    # clusterIDsOx[np.argmax(aveProbOx)] = categories[i]

print('Identified clusters Mg: ', clusterIDsMg)
print('Identified clusters Si: ', clusterIDsSi)
print('Identified clusters Ox: ', clusterIDsOx)
for i in range(len(clusterIDsMg)):
    if clusterIDsMg[i] == '':
        print('Not all clusters have been recognized!')
        exit()
for i in range(len(clusterIDsSi)):
    if clusterIDsSi[i] == '':
        print('Not all clusters have been recognized!')
        exit()
for i in range(len(clusterIDsOx)):
    if clusterIDsOx[i] == '':
        print('Not all clusters have been recognized!')
        exit()

#Save successfully trained model
if not os.path.exists('0_GMModel/Cutoff/weightsMg.npy'):
    os.makedirs('0_GMModel/Cutoff/', exist_ok=True)
    np.save(dirModel+'weightsMg.npy', modelMg.weights_, allow_pickle=False)
    np.save(dirModel+'weightsSi.npy', modelSi.weights_, allow_pickle=False)
    np.save(dirModel+'weightsOx.npy', modelOx.weights_, allow_pickle=False)
    np.save(dirModel+'meansMg.npy', modelMg.means_, allow_pickle=False)
    np.save(dirModel+'meansSi.npy', modelSi.means_, allow_pickle=False)
    np.save(dirModel+'meansOx.npy', modelOx.means_, allow_pickle=False)
    np.save(dirModel+'covariancesMg.npy', modelMg.covariances_, allow_pickle=False)
    np.save(dirModel+'covariancesSi.npy', modelSi.covariances_, allow_pickle=False)
    np.save(dirModel+'covariancesOx.npy', modelOx.covariances_, allow_pickle=False)


# Compute probabilities for each atom and write to output file
def computeProbabilityPerAtom(input_file):
    # Define output file name
    output_file = input_file.replace(".Q", ".PROB")
    output_file = output_file.replace('QValues', 'Output')
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print(output_file)
    # Read q values from input files
    pos=np.genfromtxt(input_file,skip_header=9)
    qmg, qsi, qox = readQ(input_file, startCol=8)
    # Predict probabilities for all atoms in the system
    probsMg = modelMg.predict_proba(qmg)
    probsSi = modelSi.predict_proba(qsi)
    probsOx = modelOx.predict_proba(qox)
    # try:
    #     probs = model.predict_proba(q)
    # except:
    #     print('Error when predicting structures')
    #     return

    # Start writing to output file
    os.system("head -n 8 "+input_file+"  > "+output_file)
    Pstring = ' '.join(['p'+categories[i] for i in range(len(categories))])
    #Get corresponding array index for probabilities
    dictMg = [None]*len(categories)
    dictSi = [None]*len(categories)
    dictOx = [None]*len(categories)
    for i in range(len(categories)):
        dictMg[i] = []
        dictSi[i] = []
        dictOx[i] = []
        for b in range(len(clusterIDsMg)):
            if clusterIDsMg[b] == categories[i]:
                dictMg[i] += [b]
        for b in range(len(clusterIDsSi)):
            if clusterIDsSi[b] == categories[i]:
                dictSi[i] += [b]
        for b in range(len(clusterIDsOx)):
            if clusterIDsOx[b] == categories[i]:
                dictOx[i] += [b]

    os.system("echo  'ITEM: ATOMS id type xu yu zu q "+Pstring+"' >> "+output_file)
    b1 = 0 #Counter for Mg
    b2 = 0 #Counter for Si
    b3 = 0 #Counter for Ox
    with open(output_file,'a') as fw:
        for i in range(len(pos)):
            # Write results in file
            if pos[i,1] == 1:
                prob = np.zeros(len(categories))
                for b in range(len(prob)):
                    for s in range(len(dictMg[b])):
                       prob[b] += probsMg[b1][dictMg[b][s]]
                PatomString = ' '.join([str(prob[b]) for b in range(len(prob))])
                b1 += 1
                if np.sum(prob) > 1.5:
                    print(categories)
                    print(dictMg)
                    print(clusterIDsMg)
                    print(probsMg[b1])
                    print(prob)
                    exit()
            elif pos[i,1] == 2:
                prob = np.zeros(len(categories))
                for b in range(len(prob)):
                    for s in range(len(dictSi[b])):
                       prob[b] += probsSi[b2][dictSi[b][s]]
                PatomString = ' '.join([str(prob[b]) for b in range(len(prob))])
                b2 += 1
            elif pos[i,1] == 3:
                prob = np.zeros(len(categories))
                for b in range(len(prob)):
                    for s in range(len(dictOx[b])):
                       prob[b] += probsOx[b3][dictOx[b][s]]
                PatomString = ' '.join([str(prob[b]) for b in range(len(prob))])
                b3 += 1
            fw.write("%g %g %g %g %g %g "%(pos[i,0],pos[i,1],pos[i,3],pos[i,4], pos[i,5], pos[i,6])+PatomString+"\n")
    # print(input_file,np.mean(D_WRZ),np.mean(D_BCT),np.mean(D_MC),np.mean(D_SC))

#Iterate through all files in Q directory
list_file = sorted(glob.glob(inDir+'*/QValues/**/*.Q.trj', recursive=True))
for input_file in list_file:
    computeProbabilityPerAtom(input_file)

#Iterate through all files in Q directory
list_file = sorted(glob.glob(inDir+'QValues/**/*.Q.trj', recursive=True))
for input_file in list_file:
    computeProbabilityPerAtom(input_file)