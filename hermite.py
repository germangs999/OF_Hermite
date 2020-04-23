# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 17:15:30 2019

[H,G] = dhtmtx(N,D,T)
% DHTMTX - discrete Hermite transform matrix
%
% Synopsis:
% H = DHTMTX(N) 
% H = DHTMTX(N,D)
% [H,G] = DHTMTX(N,D)
% [H,G] = DHTMTX(N,D,T)
% 
% Inputs:
% N - transform order. N+1 equals the filters length (N >= 0)
%     It has been shown in the literature that a limiting process 
%     turns the DHT into the continous Hermite transform,
%     which is based in gaussian derivatives of a given scale.
%     The scale parameter is tipically one half of the variance
%     of the Gaussian window, which is approximated by N/4.
% D - maximum order of the expanssion (0 <= D <= N)
% T - sampling distance (1 <= T <= N)
%
% Outputs:
% H - transform matrix so that H(:,k) is the filter function of order k-1
% G - interpolation matrix so that G(:,k) is the interpolating function of order k-1


%CONVMTX Convolution matrix.
%   A = CONVMTX(H,N) returns the convolution matrix, A, such that
%   convolution of H and another vector, X, of length N with the same
%   vector orientation as H may be expressed by matrix multiplication:
%   When H and X are column vectors, A*X is the same as CONV(H,X);
%   when H and X are row vectors, X*A is the same as CONV(H,X).
%
%   % Example: 
%   %   Generate a simple convolution matrix.
%
%   h = [1 2 3 2 1];
%   convmtx(h,7)        % Convolution matrix
%
%   See also CONV.


@author: German
"""
import numpy as np
from scipy import signal, misc
import math as mt


def dhtmx(N,D,T):

    #import numpy as np
    #from scipy import signal, misc
    #import math as mt
    #
    #img1_hr = np.load('img1_hr.npy')
    #N = 4;
    #D = 4;
    #T = 1;
    
    #Definir las máscaras binomiales
    if D > 0:
      B = np.array([[1, 1], [1,-1]])
    else:
      B = np.array([[1], [1]])
      
    H = B  
    
    for m in range(2,N+1):
        ####print(m)
        if H.shape[1]<= D: 
            aa = np.swapaxes(B[:,0][np.newaxis],0,1)
            aaa= np.array([[1]])
            HH =signal.convolve2d(aa,H, mode='full')
            aa2 = np.swapaxes(B[:,1][np.newaxis],0,1)
            hhh = np.swapaxes(H[:,-1][np.newaxis],0,1)
            HH2 =signal.convolve2d(aa2,hhh, mode='full')
            H = np.concatenate((HH,HH2), axis=1)
            #print(H)
            #signal.convolve2d(aa,aaa, boundary='symm', mode='same')
        else:
            aa = np.swapaxes(B[:,0][np.newaxis],0,1)
            aaa= np.array([[1]])
            H =signal.convolve2d(aa,H, mode='full')
            #print(H)
       
    C = (np.transpose(np.sqrt(H[0:D+1,0] )))/(np.power(2, N))
    H = H*np.tile(C,[N+1,1])
    #print(H)
    #####FALTAN TRES LINEAS DE CODIGO
    W = convmtx(H[:,0],N+1)
    W = np.transpose(W[range(N-np.floor(N/T).astype('int')*T,2*N+1,T),:].sum(axis=0))
    G = np.flip(H,0)/np.matmul(W[:,np.newaxis],np.ones((1,D+1)))
    return H,G
    #print(G)
    
    
def convmtx(h,n):
#    h = h[:,0]
    hh = h[:,np.newaxis]
    columna = np.concatenate((hh, np.zeros((n-1,1))), axis=0)
    W = columna
    for idx in range(1,n):
        W = np.concatenate((W, np.roll(columna, idx)), axis=1)        
    #print(W)
    return W


def dht2(X,N,D,T):
    [H, G] = dhtmx(N,D,T)
    i=1
    
    for n in range(0, min((D+1,2*N+1))):
        for m in range(max(0,n-(N+1)),min((N+1),n+1)):
            i=i+1        
    # print(i)
            
    Y=np.zeros([np.ceil(X.shape[0]/T).astype('int'), np.ceil(X.shape[1]/T).astype('int'), i-1])
    #En el ejemplo n va de 0 a 4, m va de 0 a 4 paso a pasito
    i=1
    for n in range(0, min((D+1,2*N+1))):
        for m in range(max(0,n-(N+1)),min((N+1),n+1)):
            vec_vert = H[:,m][np.newaxis]
            vec_hor = H[:,n-m][np.newaxis]
            matpol = np.matmul(np.transpose(vec_vert), vec_hor)
            #y =signal.convolve2d(X/255,matpol, mode='same')
            y =signal.convolve2d(X,matpol, mode='same')#Fines de comparación
            Y[:,:,i-1] = y[::T,::T]
            i=i+1
    return Y
   
    