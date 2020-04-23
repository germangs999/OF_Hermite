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
import time
from hermite import  dht2
from procesoResolucion import resolutionProcess
from HermiteRotado import principal_HR_mod
from warping import of_warp_multi_mod
from calculoColor import computeColor
from scipy.interpolate import RegularGridInterpolator
import scipy.io as sio
import threading
from joblib import Parallel, delayed

####Definir las constantes
alpha=10; #100 sua  500   90    para el ground true en ultrasonido 10  7 % peso del funcional
gamma=1;   #%int. cte. 600  10   para el ground true en ultrasonido 1	 % peso del funcional
num_levels=1; #%40               =1  ;optimos para sinteticas3 %escalas
omega =1.8
truerange = 3;   
range1 = truerange * 1.04;

###Definir variables
itera_oter=[5]#,10];#%20               =3  ;optimos para sinteticas3 %iteraciones del programa, minimo 2
itera_iner=[5]#,10];#%#50               =30 ;funciÃ³n de error, te permite corregir ciertas cosas en los resultdos como indeterminaciones
sg = np.array([1.1])#0.68, 1.1, 1.7, 2.3]) ##Desviaciones estandar        
rotacion =1
    
volumen = io.imread('patient0001_4CH_sequence.mhd', plugin='simpleitk')
#Reducir el tamaño del volumen a la final
volumen = volumen[:,0:volumen.shape[1]:2,0:volumen.shape[2]:2]

for isg in range(sg.shape[0]):
    ti = time.time()
    volumenU = np.zeros([volumen.shape[0], volumen.shape[1], volumen.shape[2]]) 
    volumenV = np.zeros([volumen.shape[0], volumen.shape[1], volumen.shape[2]])
    ##Paralelización del calculo de las velocidades
    results = Parallel(n_jobs=mp.cpu_count(), prefer='threads')(delayed(of_warp_multi_mod)(volumen[idv-volumen.shape[0],:,:], volumen[idv+1-volumen.shape[0],:,:], alpha, gamma, num_levels, itera_oter[0], itera_iner[0], rotacion,sg[0]) for idv in range(volumen.shape[0]))
    [m, n] = volumen[0,:,:].shape
    for idv in range(volumen.shape[0]):
        volumenU[idv,:,:] = cv2.resize(results[idv][0], (n,m), interpolation = cv2.INTER_CUBIC)
        volumenV[idv,:,:] = cv2.resize(results[idv][1], (n,m), interpolation = cv2.INTER_CUBIC)
    del results
    sio.savemat('patient0001_UV',{'volumen':volumen.astype('uint8'), 'u2':volumenU.astype('float32'), 'v2':volumenV.astype('float32')})
    print(str(time.time()-ti))       

