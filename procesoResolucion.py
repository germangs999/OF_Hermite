# -*- coding: utf-8 -*-
"""
Created on Fri Nov 29 13:16:20 2019
fucion resolutionProcess

[du dv]=resolutionProcess(img1,img2, IH1,IH2,alpha,gamma,omega,uinit,vinit,outer_iter,inner_iter,hhh)
img1 es la imagen original 1
img2 es la imagen original 1
IH1 es el conjunto de img1 filtradas
IH2 es el conjunto de img2 filtradas


@author: German
"""
from scipy.signal import convolve
from scipy.sparse import csr_matrix, tril, triu, spdiags, linalg, save_npz, load_npz
import matplotlib.pylab as plt
import numpy as np
import numpy.matlib

def funang(x,y,theta):
    m =x
    n=y+m
    c=np.power((np.cos(theta)),m);
    s= np.power((np.sin(theta)),(n-m))
    k = np.sqrt(np.math.factorial(n) / (np.math.factorial(m)*np.math.factorial(n-m)))
    PSI = k*c*s
    return PSI

def fspecial_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def psiDerivative(x):
    epsilon = 1e-2
    y = np.maximum(np.minimum(2*np.sqrt(x), (1/epsilon)*np.ones_like(x)), (epsilon)*np.ones_like(x))
    y = 1/y
    return y
def computePsidashFS(U,V):
    [h,w]=U.shape
    psidashFS=np.zeros([2*h + 1,2*w + 1])
    ux = convolve(U,np.array([1, -1])[np.newaxis,:])
    uy = convolve(U,np.transpose(np.array([1, -1])[np.newaxis,:]))
    vx = convolve(V,np.array([1, -1])[np.newaxis,:])
    vy = convolve(V,np.transpose(np.array([1, -1])[np.newaxis,:]))
    #Otras convoluciones
    uxd = convolve(ux,np.array([1, 1])[np.newaxis,:]/2, 'valid')
    vxd = convolve(vx,np.array([1, 1])[np.newaxis,:]/2, 'valid')
    ##Ahora en y
    uyd = convolve(uy,np.transpose(np.array([1, 1])[np.newaxis,:]/2),'valid')
    vyd = convolve(vy,np.transpose(np.array([1, 1])[np.newaxis,:]/2),'valid')
    #Suma de potencias
    t = convolve(uyd,np.array([1, 1])[np.newaxis,:]/2)
    uxpd = np.power(ux,2)+np.power(t,2)
    t = convolve(uxd,np.transpose(np.array([1, 1])[np.newaxis,:]/2))
    uypd = np.power(uy,2)+np.power(t,2)
    t = convolve(vyd,np.array([1, 1])[np.newaxis,:]/2)
    vxpd = np.power(vx,2)+np.power(t,2)
    t = convolve(vxd,np.transpose(np.array([1, 1])[np.newaxis,:]/2))
    vypd = np.power(vy,2)+np.power(t,2)
    
    hola = psiDerivative(uypd + vypd)
    adios = psiDerivative(uxpd + vxpd)
    #rellenar1
    xx=yy=0
    for x in range(0,psidashFS.shape[0],2):
        for y in range(1,psidashFS.shape[1],2):
            psidashFS[x,y] =hola[xx,yy]
            #print(yy)
            yy=yy+1
        yy=0
        #print(xx)
        xx=xx+1    
    #rellenar 2
    xx=yy=0
    for x in range(1,psidashFS.shape[0],2):
        for y in range(0,psidashFS.shape[1],2):
            psidashFS[x,y] =adios[xx,yy]
            #print(yy)
            yy=yy+1
        yy=0
        #print(xx)
        xx=xx+1
    return psidashFS    

def constructMatrix(img1,img2,IH1,IH2,pd,pdfs,u,v,gamma,hhh,CCC):
    I00=IH1[:,:,0];
    I10=IH1[:,:,1];
    I01=IH1[:,:,2];
    I20=IH1[:,:,3];
    I11=IH1[:,:,4];
    I02=IH1[:,:,5];
    I30=IH1[:,:,6];
    I21=IH1[:,:,7];
    I12=IH1[:,:,8];
    I03=IH1[:,:,9];    
    
    I00w=IH2[:,:,0];
    I10w=IH2[:,:,1];
    I01w=IH2[:,:,2];
    I20w=IH2[:,:,3];
    I11w=IH2[:,:,4];
    I02w=IH2[:,:,5];
    I30w=IH2[:,:,6];
    I21w=IH2[:,:,7];
    I12w=IH2[:,:,8];
    I03w=IH2[:,:,9];
    
    thetaw = np.arctan2(IH2[:,:,2], IH2[:,:,1]) 
    theta = np.arctan2(IH1[:,:,2], IH1[:,:,1]) 
    
    P10w=funang(1, 0, thetaw);
    P01w=funang(0, 1, thetaw);
    P20w=funang(2, 0, thetaw);
    P11w=funang(1, 1, thetaw);
    P02w=funang(0, 2, thetaw);
    P30w=funang(3, 0, thetaw);
    P21w=funang(2, 1, thetaw);
    P12w=funang(1, 2, thetaw); 
    P03w=funang(0, 3, thetaw);
    
    P10=funang(1, 0, theta);
    P01=funang(0, 1, theta);
    P20=funang(2, 0, theta);
    P11=funang(1, 1, theta);
    P02=funang(0, 2, theta);
    P30=funang(3, 0, theta);
    P21=funang(2, 1, theta);
    P12=funang(1, 2, theta);
    P03=funang(0, 3, theta);
    
    R1Tw=I01w*P01w + I10w*P10w
    R2Tw=I02w*P02w + I11w*P11w + I20w*P20w
    R3Tw=0;
    
    R1T=I01*P01 + I10*P10;
    R2T=I02*P02 + I11*P11 + I20*P20;
    R3T=0;
    
    R1Tm1w=I11w*P01w + I20w*P10w;
    R2Tm1w=I12w*P02w + I21w*P11w + I30w*P20w;
    R3Tm1w=0;
    
    R1Tn1w=I02w*P01w + I11w*P10w;
    R2Tn1w=I03w*P02w + I12w*P11w + I21w*P20w;
    R3Tn1w=0;
    
    R1T=I01*P01 + I10*P10;
    R2T=I02*P02 + I11*P11 + I20*P20;
    R3T=0;
    
    R1Tm1w1=I11*P01 + I20*P10;
    R2Tm1w1=I12*P02 + I21*P11 + I30*P20;
    R3Tm1w1=0;
    
    R1Tn1w1=I02*P01 + I11*P10;
    R2Tn1w1=I03*P02 + I12*P11 + I21*P20;
    R3Tn1w1=0;
    
    Imw=R1Tw+R2Tw+R3Tw;
    Im=R1T+R2T+R3T;
    
    Im1w=R1Tm1w+R2Tm1w+R3Tm1w;
    In1w=R1Tn1w+R2Tn1w+R3Tn1w;
    
    Im1w1=R1Tm1w1+R2Tm1w1+R3Tm1w1;
    In1w1=R1Tn1w1+R2Tn1w1+R3Tn1w1;
    
    [ht,wt]=u.shape
    pdfs[0,:]=0;
    pdfs[:,0]=0;
    pdfs[-1,:]=0;
    pdfs[:,-1]=0;
    #Matriz de 6 *5000 (hay que ver qué pasa)
    tmp=np.matlib.repmat(range(1,2*ht*wt+1),6,1)
    rs = np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    cs=np.copy(rs)
    ss=np.zeros(rs.shape) #%Valores
    cs[range(0,cs.shape[0],6),:]=rs[range(0,rs.shape[0],6),:] - 2*ht#x-1
    cs[range(1,cs.shape[0],6),:]=rs[range(1,rs.shape[0],6),:] - 2 #y-1
    cs[range(8,cs.shape[0],12),:]=rs[range(8,rs.shape[0],12),:] - 1 #v
    cs[range(3,cs.shape[0],12),:]=rs[range(3,rs.shape[0],12),:] + 1 #u
    cs[range(4,cs.shape[0],6),:]=rs[range(4,rs.shape[0],6),:] + 2 #y+1
    cs[range(5,cs.shape[0],6),:]=rs[range(5,rs.shape[0],6),:] + 2*ht #x+1
    
    #pdfsum=pdfs(1:2:2*ht, 2:2:end) + pdfs(3:2:end, 2:2:end)+...
    #		 pdfs(2:2:end, 1:2:2*wt) + pdfs(2:2:end, 3:2:end)
    pdfsum = pdfs[0:2*ht:2, 1:pdfs.shape[1]:2]+pdfs[2:pdfs.shape[0]:2, 1:pdfs.shape[1]:2]+pdfs[1:pdfs.shape[0]:2, 0:2*wt:2]+pdfs[1:pdfs.shape[0]:2, 2:pdfs.shape[1]:2]
    uapp=pd*(np.power(I10w,2) + gamma*(np.power(Im1w,2))) + pdfsum+CCC*1 #C es el valor de suavisado 1,10,100 entre mayor mas suavisado
    vapp=pd*(np.power(I01w,2) + gamma*(np.power(In1w,2))) + pdfsum+CCC*1
    uvapp=pd*(I10w*I01w + gamma*(Im1w*In1w))
    vuapp=pd*(I01w*I10w + gamma*(In1w*Im1w))
    
    ss[2:ss.shape[0]:12,:] = np.reshape(np.transpose(uapp), (uapp.shape[0]*uapp.shape[1],1))
    ss[9:ss.shape[0]:12,:] = np.reshape(np.transpose(vapp), (vapp.shape[0]*vapp.shape[1],1))
    ss[3:ss.shape[0]:12,:] = np.reshape(np.transpose(uvapp), (uvapp.shape[0]*uvapp.shape[1],1))
    ss[8:ss.shape[0]:12,:] = np.reshape(np.transpose(vuapp), (vuapp.shape[0]*vuapp.shape[1],1))
    
    #u(j) ecuaciones de EULER
    tmp=pdfs[1:pdfs.shape[0]:2, 0:2*wt:2]
    ss[0:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    ss[6:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    tmp=pdfs[1:pdfs.shape[0]:2, 2:pdfs.shape[1]:2]
    ss[5:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    ss[11:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    
    #v(j) ecuaciones de EULER
    tmp=pdfs[0:2*ht:2, 1:pdfs.shape[1]:2]
    ss[1:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    ss[7:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    tmp=pdfs[2:pdfs.shape[0]:2, 1:pdfs.shape[1]:2]
    ss[4:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    ss[10:ss.shape[0]:12,:] = -np.reshape(np.transpose(tmp), (tmp.shape[0]*tmp.shape[1],1))
    
    upad=np.pad(u, (1, 1), 'constant', constant_values=(0))
    vpad=np.pad(v, (1, 1), 'constant', constant_values=(0))
    
    #Terminos constantes primer ecuacion Euler.
    pdsum1 = pdfs[1:pdfs.shape[0]:2, 0:2*wt:2]*(upad[1:ht+1:, 0:wt:]- upad[1:ht+1:, 1:wt+1:])            
    pdsum2 = pdfs[1:pdfs.shape[0]:2, 2:pdfs.shape[1]:2]*(upad[1:ht+1:, 2:upad.shape[1]:]- upad[1:ht+1:, 1:wt+1:])
    pdsum3 = pdfs[0:2*ht:2, 1:pdfs.shape[1]:2]*(upad[0:ht:, 1:wt+1:]- upad[1:ht+1:, 1:wt+1:])
    pdsum4 = pdfs[2:pdfs.shape[0]:2, 1:pdfs.shape[1]:2]*(upad[2:upad.shape[0]:, 1:wt+1:]- upad[1:ht+1:, 1:wt+1:])   
    pdfaltsumu = pdsum1 + pdsum2 + pdsum3 + pdsum4
    
    pdsum1 = pdfs[1:pdfs.shape[0]:2, 0:2*wt:2]*(vpad[1:ht+1:, 0:wt:]- vpad[1:ht+1:, 1:wt+1:])            
    pdsum2 = pdfs[1:pdfs.shape[0]:2, 2:pdfs.shape[1]:2]*(vpad[1:ht+1:, 2:vpad.shape[1]:]- vpad[1:ht+1:, 1:wt+1:])
    pdsum3 = pdfs[0:2*ht:2, 1:pdfs.shape[1]:2]*(vpad[0:ht:, 1:wt+1:]- vpad[1:ht+1:, 1:wt+1:])
    pdsum4 = pdfs[2:pdfs.shape[0]:2, 1:pdfs.shape[1]:2]*(vpad[2:vpad.shape[0]:, 1:wt+1:]- vpad[1:ht+1:, 1:wt+1:])   
    pdfaltsumv = pdsum1 + pdsum2 + pdsum3 + pdsum4
    
    # el que estaba v1
    constu=pd*(I10w*(I00w - I00) + gamma*(Im1w*(Imw-Im))) - pdfaltsumu
    constv=pd*(I01w*(I00w - I00) + gamma*(In1w*(Imw-Im))) - pdfaltsumv
    
    b=np.zeros([2*ht*wt,1])
    b[0:b.shape[0]:2,:]=-np.reshape(np.transpose(constu), (constu.shape[0]*constu.shape[1],1))
    b[1:b.shape[0]:2,:]=-np.reshape(np.transpose(constv), (constv.shape[0]*constv.shape[1],1))
    
    [ind, _] = np.where(cs > 0)
    rs1=rs[ind];
    cs1=cs[ind];
    ss1=ss[ind];
    
    [ind1, _] = np.where(cs1 <(2*ht*wt + 1))
    rs2=rs1[ind1];
    cs2=cs1[ind1];
    ss2=ss1[ind1];
    
    #rs2 = np.squeeze(rs2.astype('int64'), axis=1)
    #cs2 = np.squeeze(cs2.astype('int64'), axis=1)
    #ss2 = np.squeeze(ss2.astype('int64'), axis=1)
    
    rs2 = np.squeeze(rs2, axis=1)
    cs2 = np.squeeze(cs2, axis=1)
    ss2 = np.squeeze(ss2, axis=1)
    
    #A = csr_matrix((rs2, (cs2, ss2)),  dtype=np.float)
    A = csr_matrix((ss2, (rs2, cs2)),  dtype=np.float)
    A.eliminate_zeros()
    A = A[1::,1::]
    return A, b

def split( A,b,w,flag):
    [m,n] = A.shape
    if (flag == 1):
        MM = A.diagonal()
        M = spdiags(MM,0,MM.shape[0],MM.shape[0]) #M.tocsr()[0:5,0:5].todense()
        N = M-A
    elif (flag == 2):
        b = w*b
        #M =  w * np.tril( A.todense(), k=-1 ) + np.diag(np.diag(A.todense()))
        MM = A.diagonal()
        M =  w *tril( A, k=-1 )+spdiags(MM,0,MM.shape[0],MM.shape[0])
        #N = -w * np.triu( A.todense(),  k=1 ) + ( 1.0 - w ) * np.diag(np.diag(A.todense()))
        N = -w *triu( A, k=1 ) + ((1-w)*spdiags(MM,0,MM.shape[0],MM.shape[0]))
    return M, N, b

def sor(A, x, b, w, max_it, tol):
#    print(max_it)
    flag = 0
    itera = 0
    bnrm2 = np.linalg.norm(b)
    if ( bnrm2 == 0.0 ):
         bnrm2 = 1.0
    r = b - A*x    
    error = np.linalg.norm(r)/bnrm2
    if ( error < tol ): 
        return x, error, itera, flag
    [ M, N, b ] = split( A, b, w, 2 ) #Matrix splitting
    for it in range(max_it):
        #print(it)
        x_1 = np.copy(x)
        den = N*x+b
        #M = M.astype('float16')
        x = linalg.spsolve(M,den)[:,np.newaxis]#linalg.spsolve(M,den)
        error = np.linalg.norm(x-x_1)/np.linalg.norm(x) #calcular erorr
        if (error <= tol):
            break
    b = b/w    
    # r = b - np.matmul(A,x)
    r = b - A*x
    if(error > tol):
        flag = 1
    return x, error, itera, flag
####################################################################################    
#La separacion de los resultados por nombres es comun si es rotada o no
def resolutionProcess(img1,img2, IH1,IH2,alpha,gamma,omega,uinit,vinit,outer_iter,inner_iter,hhh):
    I00=IH1[:,:,0];
    I10=IH1[:,:,1];
    I01=IH1[:,:,2];
    I20=IH1[:,:,3];
    I11=IH1[:,:,4];
    I02=IH1[:,:,5];
    I30=IH1[:,:,6];
    I21=IH1[:,:,7];
    I12=IH1[:,:,8];
    I03=IH1[:,:,9];
    
    I00w=IH2[:,:,0];
    I10w=IH2[:,:,1];
    I01w=IH2[:,:,2];
    I20w=IH2[:,:,3];
    I11w=IH2[:,:,4];
    I02w=IH2[:,:,5];
    I30w=IH2[:,:,6];
    I21w=IH2[:,:,7];
    I12w=IH2[:,:,8];
    I03w=IH2[:,:,9];
    
    thetaw = np.arctan2(IH2[:,:,2], IH2[:,:,1]) 
    theta = np.arctan2(IH1[:,:,2], IH1[:,:,1]) 
    
    P10w=funang(1, 0, thetaw);
    P01w=funang(0, 1, thetaw);
    P20w=funang(2, 0, thetaw);
    P11w=funang(1, 1, thetaw);
    P02w=funang(0, 2, thetaw);
    P30w=funang(3, 0, thetaw);
    P21w=funang(2, 1, thetaw);
    P12w=funang(1, 2, thetaw); 
    P03w=funang(0, 3, thetaw);
    
    P10=funang(1, 0, theta);
    P01=funang(0, 1, theta);
    P20=funang(2, 0, theta);
    P11=funang(1, 1, theta);
    P02=funang(0, 2, theta);
    P30=funang(3, 0, theta);
    P21=funang(2, 1, theta);
    P12=funang(1, 2, theta);
    P03=funang(0, 3, theta);
    
    I10a = np.empty([I10.shape[0],I10.shape[1],2])
    I10a[:,:,0] = I10
    I10a[:,:,1] = I10w
    
    S=np.array([0.0833, -0.6667, 0, 0.6667, -0.0833])[np.newaxis,:] #S = [1 0 -1]./2;
    #in the temporal domain
    St = np.array([1, -1]);
    
    #Generar un fitro gaussiana
    G = fspecial_gauss2D((3,1),sigma=1)
    n = G.shape[0]
    
    Lx = convolve(I10a,np.reshape(G, (1,1,n)),'same');
    Lx = convolve(Lx,np.reshape(G, (1,n,1)),'same');
    Lx = convolve(Lx,np.reshape(G, (n,1,1)),'same');
    
    ###Ahora para y
    I01a = np.empty([I01.shape[0],I01.shape[1],2])
    I01a[:,:,0] = I01
    I01a[:,:,1] = I01w
    Ly = convolve(I01a,np.reshape(G, (1,1,n)),'same');
    Ly = convolve(Ly,np.reshape(G, (1,n,1)),'same');
    Ly = convolve(Ly,np.reshape(G, (n,1,1)),'same');
    
    ##otra dimension???
    I00a = np.empty([I00.shape[0],I00.shape[1],2])
    I00a[:,:,0] = I00
    I00a[:,:,1] = I00w
    St =np.array([1, -1])[np.newaxis,:]
    Lot = convolve(I00a,np.reshape(St,(1,1,2)),'valid');
    Lot = convolve(Lot,np.reshape(G, (1,1,n)),'same');
    Lot = convolve(Lot,np.reshape(G, (1,n,1)),'same');
    Lot = convolve(Lot,np.reshape(G, (n,1,1)),'same');
    Lt = np.empty([Lot.shape[0],Lot.shape[1],2])
    Lt[:,:,0] = Lot[:,:,0]
    Lt[:,:,1] = Lot[:,:,0]
    
    #Me quede en la 179
    R1Tw=I01w*P01w + I10w*P10w
    R1Tw1=I01w*-P10w + I10w*P01w #No se usa depsues
    
    R2Tw=I02w*P02w + I11w*P11w + I20w*P20w
    R2Tw1=I02w*P20w - I11w*P11w + I20w*P02w #no se usa después
    
    R3Tw=0
    
    R1T=I01*P01 + I10*P10
    R1T1=I01*-P10 + I10*P01
    
    R2T=I02*P02 + I11*P11 + I20*P20
    
    R3T=0
    
    R1Tm1w=I11w*P01w + I20w*P10w
    R2Tm1w=I12w*P02w + I21w*P11w + I30w*P20w
    R3Tm1w=0
    
    R1Tn1w=I02w*P01w + I11w*P10w
    R2Tn1w=I03w*P02w + I12w*P11w + I21w*P20w
    R3Tn1w=0
    
    R1T=I01*P01 + I10*P10
    R2T=I02*P02 + I11*P11 + I20*P20
    R3T=0
    
    R1Tm1w1=I11*P01 + I20*P10;
    R2Tm1w1=I12*P02 + I21*P11 + I30*P20
    R3Tm1w1=0
    
    R1Tn1w1=I02*P01 + I11*P10;
    R2Tn1w1=I03*P02 + I12*P11 + I21*P20
    R3Tn1w1=0
    
    Imw=R1Tw+R2Tw+R3Tw
    Im=R1T+R2T+R3T
    
    Im1w=R1Tm1w+R2Tm1w+R3Tm1w
    In1w=R1Tn1w+R2Tn1w+R3Tn1w
        
    Im1w1=R1Tm1w1+R2Tm1w1+R3Tm1w1
    In1w1=R1Tn1w1+R2Tn1w1+R3Tn1w1
    
    u = uinit
    v = vinit
    
    [ht, wt]=I00.shape;
    
    du =np.zeros([ht,wt])
    dv =np.zeros([ht,wt])
    tol=1e-8*np.ones([2*ht*wt,1])
    
    #Solucion para las iteraciones SOR
    duv=np.zeros([2*ht*wt,1]);
    
    #Hay varias funciones para definir C
    eps = np.spacing(1)
    CCC=(1/(np.sqrt(np.power((I10w-I10),2)+np.power((I01w-I01),2)+np.power((I00w-I00),2))+eps))*0; #no tiene sentido 
    
    for i in range(outer_iter):
#        print(i) #Será un cero
        psidash=psiDerivative(np.power((I00w - I00 + (I10w)*du + (I01w)*dv), 2) + gamma*(np.power((Imw - Im +  Im1w*du + In1w*dv), 2)))
        #mequede e la 296 en la funcion computePsiDashFS
        psidashFS=computePsidashFS(u + du,v + dv)
        #me quede en construct matrix
        hhh=1
        [A, b]=constructMatrix(img1,img2,IH1,IH2,psidash,alpha*psidashFS,du,dv,gamma,hhh,CCC)
        [duv, err, it, flag]=sor(A,duv,b,omega,inner_iter,tol[0,0])
        du[::,::] = np.transpose(np.reshape(duv[0:duv.shape[0]:2],(du.shape[1], du.shape[0])))
        dv[::,::] = np.transpose(np.reshape(duv[1:duv.shape[0]:2],(du.shape[1], du.shape[0])))
        
    return du, dv
