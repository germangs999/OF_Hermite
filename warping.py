# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 18:01:04 2019

@author: germa
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
from hermite import  dht2
from procesoResolucion import resolutionProcess
from scipy.interpolate import RegularGridInterpolator
from HermiteRotado import principal_HR_mod, principal_HR

#function warpedim=warping(im1,u,v)
def warping(im1,u,v):
    [h,w] = im1.shape
    nx, ny = (w, h)
    x = np.linspace(1, w, nx)
    y = np.linspace(1, h, ny)
    V = (np.squeeze(im1)).astype('float64')
    uc, vc = np.meshgrid(x, y)
    uc1=uc + u
    vc1=vc + v
    Xq = np.reshape(np.transpose(uc1), (uc1.shape[0]*uc1.shape[1],1))
    Yq = np.reshape(np.transpose(vc1), (vc1.shape[0]*vc1.shape[1],1))
    my_interpolating_function = RegularGridInterpolator((x,y), V.T, bounds_error=False)
    mat = np.concatenate((Xq,Yq), axis = 1)
    zq = my_interpolating_function(mat)
    warpedim = np.reshape(zq,(V.shape[1], V.shape[0])).T
    return warpedim

def of_warp_multi(img1,img2,alpha,gamma,num_levels,itera_outer,itera_iner, rotacion):
    omega = 1.8
    # resize image
    scale_percent = np.power(1,  num_levels) 
    width1 = int(img1.shape[1] * scale_percent)
    height1 = int(img1.shape[0] * scale_percent)
    dim1 = (width1, height1)
    img1_hr = cv2.resize(img1, dim1, interpolation = cv2.INTER_LINEAR)
    width2 = int(img2.shape[1] * scale_percent)
    height2 = int(img2.shape[0] * scale_percent)
    dim2 = (width2, height2)
    img2_hr = cv2.resize(img2, dim2, interpolation = cv2.INTER_LINEAR)
    
    #Definir u y v
    u = np.zeros(img1_hr.shape)
    v = np.zeros(img1_hr.shape)
    #print(u.shape)
    #print(v.shape)
    #Rellenar la matriz
    #rotacion = 0 #Rotacion 1 sin rotacion 0 (variable hhh)
    
    for i in range(num_levels-1,-1,-1):
        #print(i)
        dim = img1_hr.shape
        contorno = np.zeros([(dim[0]+4),(dim[1]+4)])
        contorno[2:-2,2:-2]=img1_hr
        #llenar filas
        contorno[0,:]=contorno[2,:]
        contorno[1,:]=contorno[2,:]
        contorno[-1,:]=contorno[-3,:]
        contorno[-2,:]=contorno[-3,:]
        #Llenar columnas
        contorno[:,0]=contorno[:,2]
        contorno[:,1]=contorno[:,2]
        contorno[:,-1]=contorno[:,-3]
        contorno[:,-2]=contorno[:,-3]
        #Rellenar esquinas
        contorno[0,0] = contorno[0,2]
        contorno[1,0] = contorno[0,2]
        contorno[-1,-1] = contorno[-1,-3]
        contorno[-1,-2] = contorno[-1,-3]
        contorno[-1, 0] = contorno[-1,2]
        contorno[-1, 1] = contorno[-1,2]
        contorno[0,-1] = contorno[0,-3]
        contorno[0,-2] = contorno[0,-3]
        
        img1_hr=np.copy(contorno)
        
        dim1 = img2.shape
        contorno1 = np.zeros([(dim1[0]+4),(dim1[1]+4)])
        contorno1[2:-2,2:-2]=img2_hr
        #llenar filas
        contorno1[0,:]=contorno1[2,:]
        contorno1[1,:]=contorno1[2,:]
        contorno1[-1,:]=contorno1[-3,:]
        contorno1[-2,:]=contorno1[-3,:]
        #Llenar columnas
        contorno1[:,0]=contorno1[:,2]
        contorno1[:,1]=contorno1[:,2]
        contorno1[:,-1]=contorno1[:,-3]
        contorno1[:,-2]=contorno1[:,-3]
        #Rellenar esquinas
        contorno1[0,0] = contorno1[0,2]
        contorno1[1,0] = contorno1[0,2]
        contorno1[-1,-1] = contorno1[-1,-3]
        contorno1[-1,-2] = contorno1[-1,-3]
        contorno1[-1, 0] = contorno1[-1,2]
        contorno1[-1, 1] = contorno1[-1,2]
        contorno1[0,-1] = contorno1[0,-3]
        contorno1[0,-2] = contorno1[0,-3]
        
        img2_hr=np.copy(contorno1)
        
        
        if rotacion == 0:
#            print('Transformada Discreta Hermite 2D')
            IH1=dht2(img1_hr,4,4,1)
            IH2=dht2(img2_hr,4,4,1)
            IH1 = IH1[2:-2,2:-2,:]
            IH2 = IH2[2:-2,2:-2,:]
#            print(IH1.shape)
#            print(IH2.shape)
            img11 = np.copy(img1)
            img22 = np.copy(img2)
        
            
        else:    
#            print('Transformada Rotada Discreta Hermite 2D')
            img11 = np.copy(img1_hr)
            img22 = np.copy(img2_hr)
            [ImaDescRot,AngTeta,IH11] = principal_HR(img11)
            [ImaDescRot,AngTeta,IH22] = principal_HR(img22)
            
            IH1 = np.zeros([IH11[0][0].shape[0], IH11[0][0].shape[1], 10])
            IH2 = np.zeros([IH22[0][0].shape[0], IH22[0][0].shape[1], 10])             
            kk =  0##Considera sólo  la primera desviación estándar
            for ii in range(10):
                IH1[:,:,ii]=IH11[kk][ii]
                IH2[:,:,ii]=IH22[kk][ii]
                #IH1[:,:,ii]=I11[:,:,ii]
                #IH2[:,:,ii]=I22[:,:,ii]
            IH1 = IH1[2:IH1.shape[0]-2, 2:IH1.shape[1]-2,: ]    
            IH2 = IH2[2:IH2.shape[0]-2, 2:IH2.shape[1]-2,: ]
#            print(IH1.shape)
#            print(IH2.shape)
        
        [du, dv]=resolutionProcess(img11,img22, IH1,IH2,alpha,gamma,omega,u,v,itera_outer,itera_iner,rotacion)
#        print(img11.shape)
#        print(u.shape)
#        print(v.shape)
#        print(du.shape)
#        print(dv.shape)
        u = u+du
        v = v+dv
        # # resize image
        scale_percent = np.power(1,  i) 
        width1 = int(img11.shape[1] * scale_percent)
        height1 = int(img11.shape[0] * scale_percent)
        dim1 = (width1, height1)
        img1_hr = cv2.resize(img11, dim1, interpolation = cv2.INTER_LINEAR)
        width2 = int(img22.shape[1] * scale_percent)
        height2 = int(img22.shape[0] * scale_percent)
        dim2 = (width2, height2)
        img2_hr = cv2.resize(img22, dim2, interpolation = cv2.INTER_LINEAR)
        
        u = cv2.resize(u, (img1_hr.shape[1], img1_hr.shape[0]), interpolation = cv2.INTER_LINEAR)
        v = cv2.resize(v, (img1_hr.shape[1], img1_hr.shape[0]), interpolation = cv2.INTER_LINEAR)
#        print(u.shape)
#        print(v.shape)
        #aplicar el warping
        img2_hr=(warping(img2_hr.astype('double'),u,v)).astype('uint8')
            
    return u, v
################El owarpmulti modificado
def of_warp_multi_mod(img1,img2,alpha,gamma,num_levels,itera_outer,itera_iner, rotacion, sg):
    omega = 1.8
    # resize image
    scale_percent = np.power(1,  num_levels) 
    width1 = int(img1.shape[1] * scale_percent)
    height1 = int(img1.shape[0] * scale_percent)
    dim1 = (width1, height1)
    img1_hr = cv2.resize(img1, dim1, interpolation = cv2.INTER_LINEAR)
    width2 = int(img2.shape[1] * scale_percent)
    height2 = int(img2.shape[0] * scale_percent)
    dim2 = (width2, height2)
    img2_hr = cv2.resize(img2, dim2, interpolation = cv2.INTER_LINEAR)
    
    #Definir u y v
    u = np.zeros(img1_hr.shape)
    v = np.zeros(img1_hr.shape)
#    print(u.shape)
#    print(v.shape)
    #Rellenar la matriz
    #rotacion = 0 #Rotacion 1 sin rotacion 0 (variable hhh)
    
    for i in range(num_levels-1,-1,-1):
#        print(i)
        dim = img1_hr.shape
        contorno = np.zeros([(dim[0]+4),(dim[1]+4)])
        contorno[2:-2,2:-2]=img1_hr
        #llenar filas
        contorno[0,:]=contorno[2,:]
        contorno[1,:]=contorno[2,:]
        contorno[-1,:]=contorno[-3,:]
        contorno[-2,:]=contorno[-3,:]
        #Llenar columnas
        contorno[:,0]=contorno[:,2]
        contorno[:,1]=contorno[:,2]
        contorno[:,-1]=contorno[:,-3]
        contorno[:,-2]=contorno[:,-3]
        #Rellenar esquinas
        contorno[0,0] = contorno[0,2]
        contorno[1,0] = contorno[0,2]
        contorno[-1,-1] = contorno[-1,-3]
        contorno[-1,-2] = contorno[-1,-3]
        contorno[-1, 0] = contorno[-1,2]
        contorno[-1, 1] = contorno[-1,2]
        contorno[0,-1] = contorno[0,-3]
        contorno[0,-2] = contorno[0,-3]
        
        img1_hr=np.copy(contorno)
        
        dim1 = img2.shape
        contorno1 = np.zeros([(dim1[0]+4),(dim1[1]+4)])
        contorno1[2:-2,2:-2]=img2_hr
        #llenar filas
        contorno1[0,:]=contorno1[2,:]
        contorno1[1,:]=contorno1[2,:]
        contorno1[-1,:]=contorno1[-3,:]
        contorno1[-2,:]=contorno1[-3,:]
        #Llenar columnas
        contorno1[:,0]=contorno1[:,2]
        contorno1[:,1]=contorno1[:,2]
        contorno1[:,-1]=contorno1[:,-3]
        contorno1[:,-2]=contorno1[:,-3]
        #Rellenar esquinas
        contorno1[0,0] = contorno1[0,2]
        contorno1[1,0] = contorno1[0,2]
        contorno1[-1,-1] = contorno1[-1,-3]
        contorno1[-1,-2] = contorno1[-1,-3]
        contorno1[-1, 0] = contorno1[-1,2]
        contorno1[-1, 1] = contorno1[-1,2]
        contorno1[0,-1] = contorno1[0,-3]
        contorno1[0,-2] = contorno1[0,-3]
        
        img2_hr=np.copy(contorno1)
        
        
        if rotacion == 0:
#            print('Transformada Discreta Hermite 2D')
            IH1=dht2(img1_hr,4,4,1)
            IH2=dht2(img2_hr,4,4,1)
            IH1 = IH1[2:-2,2:-2,:]
            IH2 = IH2[2:-2,2:-2,:]
#            print(IH1.shape)
#            print(IH2.shape)
            img11 = np.copy(img1)
            img22 = np.copy(img2)
        
            
        else:    
#            print('Transformada Rotada Discreta Hermite 2D')
            img11 = np.copy(img1_hr)
            img22 = np.copy(img2_hr)
            [ImaDescRot,AngTeta,IH11] = principal_HR_mod(img11,sg)
            [ImaDescRot,AngTeta,IH22] = principal_HR_mod(img22,sg)
            
            IH1 = np.zeros([IH11[0].shape[0], IH11[0].shape[1], 10])
            IH2 = np.zeros([IH22[0].shape[0], IH22[0].shape[1], 10])             
            #kk =  0##Considera sólo  la primera desviación estándar
            for ii in range(10):
                IH1[:,:,ii]=IH11[ii]#IH11[kk][ii]
                IH2[:,:,ii]=IH22[ii]#IH22[kk][ii]
                #IH1[:,:,ii]=I11[:,:,ii]
                #IH2[:,:,ii]=I22[:,:,ii]
            IH1 = IH1[2:IH1.shape[0]-2, 2:IH1.shape[1]-2,: ]    
            IH2 = IH2[2:IH2.shape[0]-2, 2:IH2.shape[1]-2,: ]
#            print(IH1.shape)
#            print(IH2.shape)
        
        [du, dv]=resolutionProcess(img11,img22, IH1,IH2,alpha,gamma,omega,u,v,itera_outer,itera_iner,rotacion)
#        print(img11.shape)
#        print(u.shape)
#        print(v.shape)
#        print(du.shape)
#        print(dv.shape)
        u = u+du
        v = v+dv
        # # resize image
        scale_percent = np.power(1,  i) 
        width1 = int(img11.shape[1] * scale_percent)
        height1 = int(img11.shape[0] * scale_percent)
        dim1 = (width1, height1)
        img1_hr = cv2.resize(img11, dim1, interpolation = cv2.INTER_LINEAR)
        width2 = int(img22.shape[1] * scale_percent)
        height2 = int(img22.shape[0] * scale_percent)
        dim2 = (width2, height2)
        img2_hr = cv2.resize(img22, dim2, interpolation = cv2.INTER_LINEAR)
        
        u = cv2.resize(u, (img1_hr.shape[1], img1_hr.shape[0]), interpolation = cv2.INTER_LINEAR)
        v = cv2.resize(v, (img1_hr.shape[1], img1_hr.shape[0]), interpolation = cv2.INTER_LINEAR)
#        print(u.shape)
#        print(v.shape)
        #aplicar el warping
        img2_hr=(warping(img2_hr.astype('double'),u,v)).astype('uint8')
            
    return u, v
