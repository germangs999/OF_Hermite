# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:06:12 2019

@author: germa
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import math
from hermite import  dht2
from procesoResolucion import resolutionProcess
from HermiteRotado import principal_HR
from warping import of_warp_multi, of_warp_multi_mod
from calculoColor import computeColor
from scipy.interpolate import RegularGridInterpolator
import time
t1 = time.time()

## Hiperparámetros para el cálculo del flujo óptico
alpha=10; #100 sua  500   90    para el ground true en ultrasonido 10  7 % peso del funcional
gamma=1;   #%int. cte. 600  10   para el ground true en ultrasonido 1	 % peso del funcional
num_levels=1; #%40               =1  ;optimos para sinteticas3 %escalas
itera_outer=5;#%20               =3  ;optimos para sinteticas3 %iteraciones del programa, minimo 2
itera_iner=5;#%#50               =30 ;funciÃ³n de error, te permite corregir ciertas cosas en los resultdos como indeterminaciones
omega =1.8
truerange = 3;   
range1 = truerange * 1.04;

#Secuencia de imágenes para calcular el flujo óptico
img1 = cv2.imread('dimetrodon10.png',cv2.COLOR_BGR2GRAY)
img2 = cv2.imread('dimetrodon11.png',cv2.COLOR_BGR2GRAY)

rotacion = 1 #Hermite clásico = 0, Hermite rotado = 1
#Cálculo del flujo óptico
u, v = of_warp_multi(img1, img2, alpha, gamma, num_levels, itera_outer, itera_iner, rotacion)
[m, n] = img1.shape
#Ajuste de las velocidades
u2 = cv2.resize(u, (n,m), interpolation = cv2.INTER_CUBIC)
v2 = cv2.resize(v, (n,m), interpolation = cv2.INTER_CUBIC)
t2 = time.time() - t1
print(t2)

#Sigue el cómputo del mapa de colores
u3 = u2/range1/math.sqrt(2)
v3 = v2/range1/math.sqrt(2)
img_fl = computeColor(u3, v3)
[height, width, tempo]=img_fl.shape
fil2 = np.round(height/2)
col2 = np.round(width/2)
x1 = np.linspace(1, width,width)
y1 = np.linspace(1, height,height)
x, y = np.meshgrid(x1, y1)
u_c = x*range1/col2 - range1
v_c = y*range1/fil2 - range1
imgcode = computeColor(u_c/truerange, v_c/truerange)

#Despliegue de los vectores de velocidad
dd = 10
plt.figure()
plt.imshow(img1,cmap="gray")
plt.axis('image')
plt.quiver(x[::dd, ::dd]-1, y[::dd, ::dd]-1, u2[::dd, ::dd], v2[::dd, ::dd], color = 'b', angles = 'xy')
