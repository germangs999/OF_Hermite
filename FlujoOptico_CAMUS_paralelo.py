# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:10:43 2020

@author: German
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 27 15:05:01 2019

@author: germa
"""
import os
import numpy as np
import skimage.io as io
import multiprocessing as mp
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import math
from hermite import  dht2
from procesoResolucion import resolutionProcess
from HermiteRotado import principal_HR_mod
from warping import of_warp_multi_mod
from calculoColor import computeColor
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio
from joblib import Parallel, delayed
import time
#import xlwt 
#from xlwt import Workbook


def correlation_coefficient(patch1, patch2):
    product = np.mean((patch1 - patch1.mean()) * (patch2 - patch2.mean()))
    stds = patch1.std() * patch2.std()
    if stds == 0:
        return 0
    else:
        product /= stds
        return product


#Funcion para encontrar el tamaño de los arreglos
def tamano_array(dir_testing,dirlist_testing):
    tamano = np.zeros([2,3])
    dirsecuencia = [ item for item in os.listdir(dir_testing+'/'+dirlist_testing) if item.endswith('sequence.mhd') ]
    for ids in range(len(dirsecuencia)):
        volumen = io.imread(dir_testing+'/'+dirlist_testing+'/'+dirsecuencia[ids], plugin='simpleitk')
        tamano[ids,:] = volumen.shape
    #print(tamano)    
    return tamano

#######################
##Definir los directorios de las bases testing y training
dir_training = 'D:/germa/DriveUP/SECTEI/BD_Challenge/training'
dir_testing = 'D:/germa/DriveUP/SECTEI/BD_Challenge/testing'

dir_results_testing = 'D:/germa/SECTEI/RESULTADOS/testing'
dir_results_training = 'D:/germa/SECTEI/RESULTADOS/training'

dirlist_training = [ item for item in os.listdir(dir_training) if os.path.isdir(dir_training+'/'+item) ]
dirlist_testing = [ item for item in os.listdir(dir_testing) if os.path.isdir(dir_testing+'/'+item) ]

####Definir las constantes
alpha=10; #100 sua  500   90    para el ground true en ultrasonido 10  7 % peso del funcional
gamma=1;   #%int. cte. 600  10   para el ground true en ultrasonido 1	 % peso del funcional
num_levels=1; #%40               =1  ;optimos para sinteticas3 %escalas
omega =1.8
truerange = 3;   
range1 = truerange * 1.04;

###Definir variables
itera_oter=[5]#,5];#5,10];#%20               =3  ;optimos para sinteticas3 %iteraciones del programa, minimo 2
itera_iner=[5]#,5];#5,10];#%#50               =30 ;funciÃ³n de error, te permite corregir ciertas cosas en los resultdos como indeterminaciones
sg = np.array([1.1])#0.68, 1.1, 1.7]) #, 2.3]) ##Desviaciones estandar del gaussiana de Hermite
rotacion = 1

for idx in range(len(dirlist_training)):
    #print(idx)
    dirsecuencia = [ item for item in os.listdir(dir_training+'/'+dirlist_training[idx]) if item.endswith('4CH_sequence.mhd') ]
    dirresultado = dir_results_training+'/'+dirlist_training[idx]
    if not(os.path.exists(dirresultado)):
        os.mkdir(dirresultado)
        
    for ids in range(len(dirsecuencia)):
        volumen = io.imread(dir_training+'/'+dirlist_training[idx]+'/'+dirsecuencia[ids], plugin='simpleitk')
        volumen = volumen[:,0:volumen.shape[1]:2,0:volumen.shape[2]:2]
        for itout in range(len(itera_oter)):   
            for itin in range(len(itera_iner)):
                if rotacion==0:
                    ti = time.time()
                    print('Paciente ', dirsecuencia[ids], 'it_out ', str(itera_oter[itout]), 'it_in ', str(itera_iner[itin]), 'rotacion ', str(rotacion))
                    volumenU = np.zeros([volumen.shape[0], volumen.shape[1], volumen.shape[2]]) 
                    volumenV = np.zeros([volumen.shape[0], volumen.shape[1], volumen.shape[2]]) 
                    results = Parallel(n_jobs=mp.cpu_count(), prefer='threads')(delayed(of_warp_multi_mod)(volumen[idv-volumen.shape[0],:,:], volumen[idv+1-volumen.shape[0],:,:], alpha, gamma, num_levels, itera_oter[itout], itera_iner[itin], rotacion,sg) for idv in range(volumen.shape[0]))
                    [m, n] = volumen[0,:,:].shape
                    for idv in range(volumen.shape[0]):
                        volumenU[idv,:,:] = cv2.resize(results[idv][0], (n,m), interpolation = cv2.INTER_CUBIC)
                        volumenV[idv,:,:] = cv2.resize(results[idv][1], (n,m), interpolation = cv2.INTER_CUBIC)
                    del results
                    cadena = dirresultado+'/'+dirsecuencia[ids][:-4] + '_ITout_'+str(itera_oter[itout])+ '_ITin_'+str(itera_iner[itin])+'_fix.mat' 
                    sio.savemat(cadena,{'volumen':volumen.astype('uint8'), 'u2':volumenU.astype('float32'), 'v2':volumenV.astype('float32')})

                    print(str(time.time()-ti))
                else:
                    for isg in range(sg.shape[0]):
                        ti = time.time()
                        print('Paciente ', dirsecuencia[ids], 'it_out ', str(itera_oter[itout]), 'it_in ', str(itera_iner[itin]), 'rotacion ', str(rotacion), 'sg ', str(sg[isg]), str(volumen.shape))
                        volumenU = np.zeros([volumen.shape[0], volumen.shape[1], volumen.shape[2]]) 
                        volumenV = np.zeros([volumen.shape[0], volumen.shape[1], volumen.shape[2]])
                        results = Parallel(n_jobs=mp.cpu_count(), prefer='threads')(delayed(of_warp_multi_mod)(volumen[idv-volumen.shape[0],:,:], volumen[idv+1-volumen.shape[0],:,:], alpha, gamma, num_levels, itera_oter[itout], itera_iner[itin], rotacion,sg[isg]) for idv in range(volumen.shape[0]))
                        [m, n] = volumen[0,:,:].shape
                        for idv in range(volumen.shape[0]):
                            volumenU[idv,:,:] = cv2.resize(results[idv][0], (n,m), interpolation = cv2.INTER_CUBIC)
                            volumenV[idv,:,:] = cv2.resize(results[idv][1], (n,m), interpolation = cv2.INTER_CUBIC)
                        del results
                        cadena =  dirresultado+'/'+dirsecuencia[ids][:-4] + '_ITout_'+str(itera_oter[itout])+ '_ITin_'+str(itera_iner[itin])+'_rotsg_'+ str(sg[isg])+'_T2.mat' 
                        sio.savemat(cadena,{'volumen':volumen.astype('uint8'), 'u2':volumenU.astype('float32'), 'v2':volumenV.astype('float32')})
                        print(str(time.time()-ti))

