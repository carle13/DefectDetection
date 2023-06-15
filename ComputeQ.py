#!/usr/bin/env python3

import copy
import pyscal.core as pc
import numpy as np
import glob
import os
import os.path
import sys

#python3 ComputeQ.py 2_Relaxation/
#python3 ComputeQ.py 0_Database/
inDir = os.path.join(sys.argv[1], '')

#List of q parameters to be computed
qOrders = [2,3,4,5,6,7,8]

#   1 ---- Mg
#   2 ---- Si
#   3 ---- O

# coordCutoff = 4.5

def computeQ(input_file):
    #Get name of output file
    output_file = input_file.replace('Input', 'QValues')
    output_file = output_file.replace(".trj", ".Q.trj")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if os.path.exists(output_file):
        print(input_file, '   Continued')
        return
    print(input_file)

    print('computing q all')
    sys = pc.System()
    sys.read_inputfile(input_file)
    sys.find_neighbors(method='voronoi')
    sys.calculate_q(qOrders, averaged=True)
    q = sys.get_qvals(qOrders, averaged=True)
    q=np.array(q)
    
    # print('computing q Mg')
    # Mg=[ i for i in sys.atoms if i.type==1]
    # sysMg=pc.System()
    # sysMg.read_inputfile(input_file)
    # sysMg.atoms=Mg
    # sysMg.find_neighbors(method='voronoi')
    # sysMg.calculate_q(qOrders, averaged=True)
    # qMg=sysMg.get_qvals(qOrders, averaged=True)
    # qMg=np.array(qMg)

    # Si=[ i for i in sys.atoms if i.type==2]
    # sysSi=pc.System()
    # sysSi.read_inputfile(input_file)
    # sysSi.atoms=Si
    # sysSi.find_neighbors(method='voronoi')
    # sysSi.calculate_q(qOrders, averaged=True)
    # qSi=sysSi.get_qvals(qOrders, averaged=True)
    # qSi=np.array(qSi)
    
    # Ox=[ i for i in sys.atoms if i.type==3]
    # sysOx=pc.System()
    # sysOx.read_inputfile(input_file)
    # sysOx.atoms=Ox
    # sysOx.find_neighbors(method='voronoi')
    # sysOx.calculate_q(qOrders, averaged=True)
    # qOx=sysOx.get_qvals(qOrders, averaged=True)
    # qOx=np.array(qOx)
    
    # b1 = 0 #Counter for Mg
    # b2 = 0 #Counter for Si
    # b3 = 0 #Counter for Ox
    pos=np.genfromtxt(input_file,skip_header=9)
    os.system("head -n 8 "+input_file+"  > "+output_file)
    os.system("echo  'ITEM: ATOMS id type x y z q2 q3 q4 q5 q6 q7 q8' >> "+output_file)
    # os.system("echo  'ITEM: ATOMS id type x y z q2 q3 q4 q5 q6 q7 q8 q2_mono q3_mono q4_mono q5_mono q6_mono q7_mono q8_mono' >> "+output_file)
    with open(output_file,'a') as fw:
        for i in np.arange(int(np.size(pos[:,0]))):
            fw.write("%g %g %g %g %g "%(pos[i,0],pos[i,1],pos[i,2],pos[i,3],pos[i,4]))
            for iq in np.arange(np.size(q[:,i])):
                fw.write("%g  "%(q[iq,i]))
            # if pos[i,1] == 1:
            #     for iq in np.arange(np.size(qMg[:,b1])):
            #         fw.write("%g  "%(qMg[iq,b1]))
            #     b1 += 1
            # elif pos[i,1] == 2:
            #     for iq in np.arange(np.size(qSi[:,b2])):
            #         fw.write("%g  "%(qSi[iq,b2]))
            #     b2 += 1
            # elif pos[i,1] == 3:
            #     for iq in np.arange(np.size(qOx[:,b3])):
            #         fw.write("%g  "%(qOx[iq,b3]))
            #     b3 += 1
            # else:
            #     print('Atom without identified type type. id: ', i)
            #     exit()
            fw.write("\n")
        # print(b1, b2, b3) 
        
# #Iterate through all lammps dump files in input directory
# list_file = sorted(glob.glob(inDir+'*/Input/**/*.trj', recursive=True))
# for input_file in list_file:
#     computeQ(input_file)

#Iterate through all lammps dump files in input directory
list_file = sorted(glob.glob(inDir+'Input/**/*.trj', recursive=True))
for input_file in list_file:
    computeQ(input_file)
