# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 21:11:05 2019

@author: germa
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import math


def makeColorwheel():
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6
    
    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros([ncols,3])
    col = 0;
    #RY
    colorwheel[0:RY,0] = 255
    colorwheel[0:RY,1] = (np.floor(255*np.array(range(0,RY))/RY)).T
    col = col+RY
    #YG
    colorwheel[col:col+YG,0] = 255 - (np.floor(255*np.array(range(0,YG))/YG)).T
    colorwheel[col:col+YG,1] = 255
    col = col+YG
    #GC
    colorwheel[col:col+GC,1] =255
    colorwheel[col:col+GC,2] = (np.floor(255*np.array(range(0,GC))/GC)).T
    col = col+GC
    #CB
    colorwheel[col:col+CB,1] = 255 - (np.floor(255*np.array(range(0,CB))/CB)).T
    colorwheel[col:col+CB,2] = 255
    col = col+CB
    #BM
    colorwheel[col:col+BM,2] =255
    colorwheel[col:col+BM,0] =(np.floor(255*np.array(range(0,BM))/BM)).T
    col = col+BM
    #MR
    colorwheel[col:col+MR,2] =255 - (np.floor(255*np.array(range(0,MR))/MR)).T
    colorwheel[col:col+MR,0] =255
    return colorwheel

def computeColor(u3,v3):
    nanIdx = np.logical_or(np.isnan(u3), np.isnan(v3))  
    u3[nanIdx] = 0
    v3[nanIdx] = 0
    colorwheel = makeColorwheel();
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.power(u3,2)+np.power(v3,2))
    a =np.arctan2(-v3, -u3)/math.pi
    fk = (a+1) /2 * (ncols-1) + 1
    k0 = np.floor(fk)
    #DUDAS
    k1 = k0+1
    k1[k1==ncols+1] = 1
    f = fk - k0
    img = np.zeros([u3.shape[0], u3.shape[1],colorwheel.shape[1]]).astype('uint8')
    for i in range(0,colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = (tmp[(k0-1).astype('int')])/255
        col1 = (tmp[(k1-1).astype('int')])/255
        col = (1-f)*col0 + f*col1 
        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])   # increase saturation with radius
        col[np.logical_not(idx)] = col[np.logical_not(idx)]*0.75
        img[:,:,i]= (np.floor(255*col*(1-nanIdx))).astype('uint8')
    return img