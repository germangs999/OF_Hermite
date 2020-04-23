# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:30:36 2020

@author: German
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 16:33:54 2020
 [ImaDescRot,AngTeta,IH11] = principal(img1);
@author: German
"""
import math
from collections import defaultdict
import time
import numpy as np

def FiltrosContinuos1D(TT, MM, ssg, tam, inF, inC, enF, enC, cord):
    LimInf = -5.3;
    LimSup =  5.3;
    t = np.linspace(LimInf, LimSup, MM+1)
#    print(t)
#    print(ssg)
    V2 = ( 1/(ssg*np.sqrt(math.pi)) )*( np.exp(-(np.power(t,2))/(np.power(ssg,2))) )
    
    n = 0;
    H0 = 1;
    DnH0 = ( (np.power(-1,n))/(np.sqrt((np.power(2,n))*np.math.factorial(n))) )* H0 * V2
    PnH0 = ( 1/(np.sqrt((np.power(2,n))*np.math.factorial(n))) ) * H0 * V2
    
    n = 1;
    H1  = 2*t/ssg;
    DnH1 = ( (np.power(-1,n))/(np.sqrt((np.power(2,n))*np.math.factorial(n))) )* H1 * V2
    PnH1 = ( 1/(np.sqrt((np.power(2,n))*np.math.factorial(n))) ) * H1 * V2
    
    n = 2;
    H2  = 4*( np.power(t/ssg,2) ) - 2; 
    DnH2 = ( (np.power(-1,n))/(np.sqrt((np.power(2,n))*np.math.factorial(n))) )* H2 * V2
    PnH2 = ( 1/(np.sqrt((np.power(2,n))*np.math.factorial(n))) ) * H2 * V2
    
    n = 3;
    H3  =  8*(np.power(t/ssg,3)) - 12*(t/ssg);
    DnH3 = ( (np.power(-1,n))/(np.sqrt((np.power(2,n))*np.math.factorial(n))) )* H3 * V2
    PnH3 = ( 1/(np.sqrt((np.power(2,n))*np.math.factorial(n))) ) * H3 * V2
    
    n = 4;
    H4  = 16*( np.power(t/ssg,4) ) - 48*( np.power(t/ssg,2)) + 12;
    DnH4 = ( (np.power(-1,n))/(np.sqrt((np.power(2,n))*np.math.factorial(n))) )* H4 * V2
    PnH4 = ( 1/(np.sqrt((np.power(2,n))*np.math.factorial(n))) ) * H4 * V2
    
    n = 5;
    H5  = 32*( np.power(t/ssg,5) ) - 160*( np.power(t/ssg,3) ) + 120*(t/ssg);
    DnH5 = ( (np.power(-1,n))/(np.sqrt((np.power(2,n))*np.math.factorial(n))) )* H5 * V2
    PnH5 = ( 1/(np.sqrt((np.power(2,n))*np.math.factorial(n))) ) * H5 * V2
    
    Dn = np.zeros([DnH5.shape[0], n+1])
    Dn[:,0] = DnH0; Dn[:,1] = DnH1; Dn[:,2] = DnH2; Dn[:,3] = DnH3; Dn[:,4] = DnH4; Dn[:,5] = DnH5;
    PnD = np.zeros([PnH5.shape[0], n+1])
    PnD[:,0] = PnH0; PnD[:,1] = PnH1; PnD[:,2] = PnH2; PnD[:,3] = PnH3; PnD[:,4] = PnH4; PnD[:,5] = PnH5;
    
    # ---------------------- Peso de la Función de Reconstrucción -------------
    if cord == 0:
        WVFi = np.concatenate((V2, np.zeros((tam[0]-1+inF).astype('int'))), axis = 0)
        WVF = np.copy(WVFi)
        for r in range(TT,WVF.shape[0],TT):
            WVFShift = np.concatenate((np.zeros(r), WVFi[0:WVFi.shape[0]-r]), axis = 0)
            WVF = WVF + WVFShift
            #print(r)
        WF = np.copy(WVF)
        WF = WF[inF.astype('int')-1:enF.astype('int')]         
        W = np.copy(WF)
    
    if cord == 1:
        WVCi = np.concatenate((V2, np.zeros((tam[1]-1+inC).astype('int'))), axis = 0)
        WVC = np.copy(WVCi)
        for r in range(TT,WVC.shape[0],TT):
            WVCShift = np.concatenate((np.zeros(r), WVCi[0:WVCi.shape[0]-r]), axis = 0)
            WVC = WVC + WVCShift
            #print(r)
        WC = np.copy(WVC)
        WC = WC[inC.astype('int')-1:enC.astype('int')]         
        W = np.copy(WC)
    return Dn, PnD, W

def FiltrosHermite2D(Dn, PnD, W):
    DtamX = Dn[0].shape[0]
    DtamY = Dn[1].shape[0]
    # -- Filtro de Orden n = 0 
    d00 = np.zeros([DtamY,DtamX])
    # -- Filtros de Orden n = 1:  --
    d10 = np.zeros([DtamY,DtamX]); 
    d01 = np.zeros([DtamY,DtamX]);
    # -- Filtros de Orden n = 2 --
    d20 = np.zeros([DtamY,DtamX]);    
    d11 = np.zeros([DtamY,DtamX]);
    d02 = np.zeros([DtamY,DtamX]);
    # -- Filtros de Orden n = 3 --
    d30 = np.zeros([DtamY,DtamX]);    
    d21 = np.zeros([DtamY,DtamX]);
    d12 = np.zeros([DtamY,DtamX]);
    d03 = np.zeros([DtamY,DtamX]);
    
    for i in range(0,DtamY):
        for j in range(0,DtamX):
            d00[i,j] = Dn[0][j,0]*Dn[1][i,0]
            
            d10[i,j] = Dn[0][j,1]*Dn[1][i,0]
            d01[i,j] = Dn[0][j,0]*Dn[1][i,1]
            
            d20[i,j] = Dn[0][j,2]*Dn[1][i,0]
            d11[i,j] = Dn[0][j,1]*Dn[1][i,1]
            d02[i,j] = Dn[0][j,0]*Dn[1][i,2]
            
            d30[i,j] = Dn[0][j,3]*Dn[1][i,0]
            d21[i,j] = Dn[0][j,2]*Dn[1][i,1]
            d12[i,j] = Dn[0][j,1]*Dn[1][i,2]
            d03[i,j] = Dn[0][j,0]*Dn[1][i,3]
            
    dD = defaultdict(dict)
    dD[0] = d00;    dD[1] = d10;    dD[2] = d01;   dD[3] = d20;   dD[4] = d11;    
    dD[5] = d02;    dD[6] = d30;    dD[7] = d21;   dD[8] = d12;   dD[9] = d03; 
    
    PtamX = PnD[0].shape[0]
    PtamY = PnD[1].shape[0]
    # -- Peso de Normalización -- 
    Wp = np.zeros([PtamY,PtamX])
    # -- Filtro de Orden n = 0 --
    p00 = np.zeros([PtamY,PtamX])
    # -- Filtros de Orden n = 1:  --
    p10 = np.zeros([PtamY,PtamX]) 
    p01 = np.zeros([PtamY,PtamX])
    # -- Filtros de Orden n = 2 --
    p20 = np.zeros([PtamY,PtamX])    
    p11 = np.zeros([PtamY,PtamX])
    p02 = np.zeros([PtamY,PtamX])
    # -- Filtros de Orden n = 3 --
    p30 = np.zeros([PtamY,PtamX]);    p21 = np.zeros([PtamY,PtamX]);
    p12 = np.zeros([PtamY,PtamX]);    p03 = np.zeros([PtamY,PtamX]);
    
    for i in range(0,PtamY):
        for j in range(0,PtamX):
            Wp[i,j] = W[0][j]*W[1][i]
            
            p00[i,j] = PnD[0][j,0]*PnD[1][i,0]/Wp[i,j]
            
            p10[i,j] = PnD[0][j,1]*PnD[1][i,0]/Wp[i,j]
            p01[i,j] = PnD[0][j,0]*PnD[1][i,1]/Wp[i,j]
            
            p20[i,j] = PnD[0][j,2]*PnD[1][i,0]/Wp[i,j]
            p11[i,j] = PnD[0][j,1]*PnD[1][i,1]/Wp[i,j]
            p02[i,j] = PnD[0][j,0]*PnD[1][i,2]/Wp[i,j]
            
            p30[i,j] = PnD[0][j,3]*PnD[1][i,0]/Wp[i,j]
            p21[i,j] = PnD[0][j,2]*PnD[1][i,1]/Wp[i,j]
            p12[i,j] = PnD[0][j,1]*PnD[1][i,2]/Wp[i,j]
            p03[i,j] = PnD[0][j,0]*PnD[1][i,3]/Wp[i,j]
    
    pR = defaultdict(dict)
    pR[0] = p00;    pR[1] = p10;    pR[2] = p01;   pR[3] = p20;   pR[4] = p11;    
    pR[5] = p02;    pR[6] = p30;    pR[7] = p21;   pR[8] = p12;   pR[9] = p03; 
    return dD, pR

def Decomposition2D(Ent, dD, tam, TT, N, inF, inC, enF, enC):
    DtamX = DtamY = DtamZ = dD[0].shape[0]
    d00 = dD[0];    d10 = dD[1];   d01 = dD[2];     d20 = dD[3];    d11 = dD[4];    
    d02 = dD[5];    d30 = dD[6];   d21 = dD[7];     d12 = dD[8];    d03 = dD[9];
    #Filtrado 3D en frecuencia
    #t = time.time()
    Ent_Freq = np.fft.fftn(Ent, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
    #elapsed = time.time() - t 
    #print(elapsed)
    
    Sal = defaultdict(dict)
    if (N<0 or N>3):
        print('No sirve')
    if N>=0 and N<4:
        #print('orden 0')
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d00
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[0] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
    if N>0 and N<4:
#        print('orden 1')
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d10
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[1] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d01
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[2] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
    if N>1 and N<4:
#        print('orden 2')
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d20
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[3] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d11
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[4] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d02
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[5] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
      
    if N>2 and N<4:
#        print('orden 3')
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d30
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[6] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d21
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[7] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d12
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[8] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
        
        FilVol = np.zeros(tam) 
        FilVol[0:DtamY, 0:DtamX]=d03
        FilVolFreq = np.fft.fftn(FilVol, s=[tam[0]+DtamY-1, tam[1]+DtamX-1])
        EntFilFreq = Ent_Freq*FilVolFreq
        S = (np.fft.ifftn(EntFilFreq)).real
        Sal[9] = S[int(inF)-1:int(enF):TT, int(inC)-1:int(enC):TT];
        del S
    return Sal

def Reconstruction2D(ImaDesc, pR, tam, T, inrF, inrC, enrF, enrC):
    PtamX = PtamY = pR[0].shape[0]
    Sal  = np.zeros([tam[0]+PtamY-1,tam[1]+PtamX-1])
    for l in range(0,len(ImaDesc)):
        if TT>1:
            print('con upsampling, este tipo de reconstruccion esta incompleta. no se debe avanzar')
            
        else:
            print('sin upsampling')  
            VolDescFreq = np.fft.fftn(ImaDesc[l],s = [tam[0]+PtamY-1,tam[1]+PtamX-1])
            FilVol = np.zeros(tam)
            FilVol[0:PtamY, 0:PtamX] = pR[l]
            FilVolFreq = np.fft.fftn(FilVol, s = [tam[0]+PtamY-1,tam[1]+PtamX-1])
            EntFilFreq = VolDescFreq*FilVolFreq
            S = (np.fft.ifftn(EntFilFreq)).real
            Sal = Sal + S
    
    SalRec = Sal[int(inrF)-1:int(enrF), int(inrC)-1:int(enrC)];
    return SalRec

def AngleEstimation2D(ImaDesc, sel):
    if sel ==0:
        AngTeta = (180/math.pi)*( -np.arctan2( ImaDesc[2], ImaDesc[1] ) )
    elif sel == 1:
        Ima10 = ImaDesc[1];
        Ima01 = ImaDesc[2];
        step_angle = 10
        angulos = np.array(range(-180,180+step_angle,step_angle))
        lon = angulos.shape[0]
        long_Mat = Ima10.shape;
        AngTeta = np.zeros(long_Mat)
        
        for y in range(0,long_Mat[0]):
            for x in range(0,long_Mat[1]):
                # c1 = np.zeros(lon)
                c1 = np.cos(np.deg2rad(angulos))*Ima10[y,x]+ np.sin(np.deg2rad(angulos))*Ima01[y,x]
                #c1.index(np.max(c1))
                i = np.argmax(np.abs(c1))
                AngTeta[y,x] = angulos[i]
    else:
        print('No existe esa opción')
    return AngTeta  

def Rotation2D(ImaDesc, N, AngTeta):
    c = np.cos(np.deg2rad(AngTeta))
    s = -np.sin(np.deg2rad(AngTeta)) 
    ImaDescRot = defaultdict(dict)
    ImaDescRot[0] = ImaDesc[0]
    ImaDescRot1 = defaultdict(dict)
    ImaDescRot1[0] = []
    if N==1:
    
        ImaDescRot[1]  =     c*ImaDesc[1]  +   s*ImaDesc[2];
        ImaDescRot[2]  =   - s*ImaDesc[1]  +   c*ImaDesc[2];
        ImaDescRot1[1]  =    c*ImaDesc[3]  +   s*ImaDesc[4];
        ImaDescRot1[2]  =  - s*ImaDesc[3]  +   c*ImaDesc[4];
    
    elif N==2:
        
        ImaDescRot[1]  =     c*ImaDesc[1]  +   s*ImaDesc[2];
        ImaDescRot[2]  =   - s*ImaDesc[1]  +   c*ImaDesc[2];
        ImaDescRot1[1]  =    c*ImaDesc[3]  +   s*ImaDesc[4];
        ImaDescRot1[2]  =  - s*ImaDesc[3]  +   c*ImaDesc[4];
        
        ImaDescRot[3]  =   (np.power(c,2))*ImaDesc[3] +        2*c*s*ImaDesc[4]  +  (np.power(s,2))*ImaDesc[5];     
        ImaDescRot[4]  = - (c*s)*ImaDesc[3] + (np.power(c,2)- np.power(s,2))*ImaDesc[4]  +  (c*s)*ImaDesc[5];     
        ImaDescRot[5]  =   (np.power(s,2))*ImaDesc[3] -        2*c*s*ImaDesc[4]  +  (np.power(c,2))*ImaDesc[5];                          
        ImaDescRot1[3]  =   (np.power(c,2))*ImaDesc[6] +        2*c*s*ImaDesc[7]  +  (np.power(s,2))*ImaDesc[8];     
        ImaDescRot1[4]  = - (c*s)*ImaDesc[6] + (np.power(c,2)- np.power(s,2))*ImaDesc[7]  +  (c*s)*ImaDesc[8];     
        ImaDescRot1[5]  =   (np.power(s,2))*ImaDesc[6] -        2*c*s*ImaDesc[7]  +  (np.power(c,2))*ImaDesc[8];
    
    elif N==3:
        
        ImaDescRot[1]  =     c*ImaDesc[1]  +   s*ImaDesc[2];
        ImaDescRot[2]  =   - s*ImaDesc[1]  +   c*ImaDesc[2];
        ImaDescRot1[1]  =    c*ImaDesc[3]  +   s*ImaDesc[4];
        ImaDescRot1[2]  =  - s*ImaDesc[3]  +   c*ImaDesc[4];
        
        ImaDescRot[3]  =   (np.power(c,2))*ImaDesc[3] +        2*c*s*ImaDesc[4]  +  (np.power(s,2))*ImaDesc[5];     
        ImaDescRot[4]  = - (c*s)*ImaDesc[3] + (np.power(c,2)- np.power(s,2))*ImaDesc[4]  +  (c*s)*ImaDesc[5];     
        ImaDescRot[5]  =   (np.power(s,2))*ImaDesc[3] -        2*c*s*ImaDesc[4]  +  (np.power(c,2))*ImaDesc[5];                          
        ImaDescRot1[3]  =   (np.power(c,2))*ImaDesc[6] +        2*c*s*ImaDesc[7]  +  (np.power(s,2))*ImaDesc[8];     
        ImaDescRot1[4]  = - (c*s)*ImaDesc[6] + (np.power(c,2)- np.power(s,2))*ImaDesc[7]  +  (c*s)*ImaDesc[8];     
        ImaDescRot1[5]  =   (np.power(s,2))*ImaDesc[6] -        2*c*s*ImaDesc[7]  +  (np.power(c,2))*ImaDesc[8];    
        
        ImaDescRot[6]  = (np.power(c,3))*ImaDesc[6] + 3*(np.power(c,2))*s*ImaDesc[7] + 3*c*(np.power(s,2))*ImaDesc[8] + (np.power(s,3))*ImaDesc[9];                                        
        ImaDescRot[7]  = - (np.power(c,2))*s*ImaDesc[6] + (np.power(c,3) - 2*c*(np.power(s,2)))*ImaDesc[7] +  (2*(np.power(c,2))*s - np.power(s,3))*ImaDesc[8] + c*(np.power(s,2))*ImaDesc[9];
        ImaDescRot[8] =  c*(np.power(s,2))*ImaDesc[6] + (np.power(s,3) - 2*(np.power(c,2))*s)*ImaDesc[7] + (np.power(c,3) - 2*c*(np.power(s,2)))*ImaDesc[8] + (np.power(c,2))*s*ImaDesc[9];
        ImaDescRot[9] =  -(np.power(s,3))*ImaDesc[6] + 3*c*(np.power(s,2))*ImaDesc[7] + 3*(np.power(c,2))*s*ImaDesc[8] + (np.power(c,3))*ImaDesc[9];
        
        ImaDescRot1[6] = ImaDescRot1[7] =ImaDescRot1[8] =ImaDescRot1[9] =[]    
        
    else:
        print('No hay implementación para esa opción')    
    return ImaDescRot,ImaDescRot1


def HermiteTransform2DFreq(Ent, TT, M, ssg, N, Sel, tam):
    # ---------------- Parámetro de Distancia entre Ventanas ------------------
    Lg = M+1
    if Lg[0]%2==0:#redundante, lo voy a dejar mientras
        inF = np.round(Lg[0]/2)
        inC = np.round(Lg[1]/2)
        enF = inF + tam[0] - 1
        enC = inC + tam[1] - 1
        inrF = np.round(Lg[0]/2)
        inrC = np.round(Lg[1]/2)
        enrF = inrF + tam[0] - 1
        enrC = inrC + tam[1] - 1
    else:
        inF = np.round(Lg[0]/2)
        inC = np.round(Lg[1]/2)
        enF = inF + tam[0] - 1
        enC = inC + tam[1] - 1
        inrF = np.round(Lg[0]/2)
        inrC = np.round(Lg[1]/2)
        enrF = inrF + tam[0] - 1
        enrC = inrC + tam[1] - 1
    
    #--Mas celdas----------------- Computing Hermite Filters 1D ------------------------
    Dn = {}#cell(1,3);
    PnD = {}#cell(1,3);
    W = {}#cell(1,3);
        
    #  FiltrosContinuos1D(T, M(cord), sg, tam, inF, inC, enF, enC, cord);
    for cord in range(0,2):
        #print(cord)
        #MM = M[cord]
        Dn[cord], PnD[cord], W[cord] = FiltrosContinuos1D(TT, M[cord], ssg, tam, inF, inC, enF, enC, cord)
    #Linea 45 de HermiteTransform2DFreq
    # -------------------- Computing Hermite Filters 3D -----------------------
    [dD, pR] = FiltrosHermite2D(Dn, PnD, W)
    
    if Sel==0:
        #print("Descomposicion")
        Sal = Decomposition2D(Ent, dD, tam, TT, N, inF, inC, enF, enC)
        return Sal, Dn
    elif Sel == 1:
        #print("Reconstruccion")
        Sal = Reconstruction2D(Ent, pR, tam, TT, inrF, inrC, enrF, enrC)
        return Sal, Dn
    elif Sel == 2:
        #print("Estimacion del angulo")
        AngTeta = AngleEstimation2D(Ent, 0)
        [Sal,Sal1] = Rotation2D(Ent, N, AngTeta)
        return Sal,Sal1,AngTeta,Dn
    else:
        print("No existe esta opcion")    
    
    





############INICIo
#[ImaDescRot,AngTeta,ImaDesc] = 
def principal_HR(Ima):
#Ima = img11
    N = 3;                # Orden de la Descomposición 
                          #Cuando es 1 los coeficientes = 00 10 01                        
    # Mx = 16; My = 16;     # Dimensión del Filtro en Cada Eje Cordenado
    Mx = 11; My = 11;     # Dimensión del Filtro en Cada Eje Cordenado   11   11
    M = np.array([Mx, My])
    
    Nc = 10                     # No. de Coeficientes de Descomposición
    Ns = 4                    # No. de Escalas de Descomposición
    #sg = [0.5 1.2 1.8 2.4 3.0];     # Desviación Estándar de la Gaussiana en cada Escala
    sg = np.array([0.68, 1.1, 1.7, 2.3]);     # Desviación Estándar de la Gaussiana en cada Escala   0.68
    #sg = [3 3 3 3 3];     # Desviación Estándar de la Gaussiana en cada Escala
    # T  = [1 2 4 8 16];             # Valor de Submuestreo para cada Escala
    T  = np.array([1, 1, 1, 1]);              # Valor de Submuestreo para cada Escala
    
    #Las cells
    ImaDesc     = defaultdict(dict)
    ImaDescRot  = defaultdict(dict)
    #ImaRotProc  = defaultdict(dict)
    AngTeta     = defaultdict(dict)
    ImaDescRot1 = defaultdict(dict)
    
    ##
    tamSub = np.zeros([Ns,2])
    tam=Ima.shape
    
    #Inicia el ciclo de las escalas
    for s in range(Ns):
        Sel = 0
        [ImaDesc[s],_] = HermiteTransform2DFreq(Ima, T[s], M, sg[s], N, Sel, tam)
        tamSub[s,:] = ImaDesc[s][0].shape
        Sel = 2
        [ImaDescRot[s],ImaDescRot1[s], AngTeta[s], Dn] =  HermiteTransform2DFreq(ImaDesc[s], T[s], M, sg[s], N, Sel, tam)
        
    return ImaDescRot,AngTeta,ImaDesc    

############ Hermite Rotado MODIFICADO
#[ImaDescRot,AngTeta,ImaDesc] = 
def principal_HR_mod(Ima,sg):
#Ima = img11
    N = 3;                # Orden de la Descomposición 
                          #Cuando es 1 los coeficientes = 00 10 01                        
    # Mx = 16; My = 16;     # Dimensión del Filtro en Cada Eje Cordenado
    Mx = 11; My = 11;     # Dimensión del Filtro en Cada Eje Cordenado   11   11
    M = np.array([Mx, My])
    
    Nc = 10                     # No. de Coeficientes de Descomposición
    #Ns = 4                    # No. de Escalas de Descomposición
    #sg = [0.5 1.2 1.8 2.4 3.0];     # Desviación Estándar de la Gaussiana en cada Escala
    #sg = np.array([0.68, 1.1, 1.7, 2.3]);     # Desviación Estándar de la Gaussiana en cada Escala   0.68
    #sg = [3 3 3 3 3];     # Desviación Estándar de la Gaussiana en cada Escala
    # T  = [1 2 4 8 16];             # Valor de Submuestreo para cada Escala
    T  = 1;              # Valor de Submuestreo para cada Escala
    
    #Las cells
    ImaDesc     = defaultdict(dict)
    ImaDescRot  = defaultdict(dict)
    #ImaRotProc  = defaultdict(dict)
    AngTeta     = defaultdict(dict)
    ImaDescRot1 = defaultdict(dict)
    
    ##
#    tamSub = np.zeros([Ns,2])
    tam=Ima.shape
    
    #Inicia el ciclo de las escalas
    #for s in range(Ns):
    Sel = 0
    [ImaDesc,_] = HermiteTransform2DFreq(Ima, T, M, sg, N, Sel, tam)
    #tamSub[s,:] = ImaDesc[s][0].shape
    Sel = 2
    [ImaDescRot,ImaDescRot1, AngTeta, Dn] =  HermiteTransform2DFreq(ImaDesc, T, M, sg, N, Sel, tam)
        
    return ImaDescRot,AngTeta,ImaDesc    




