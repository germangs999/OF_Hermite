# Flujo Óptico con Transformada Hermite

Este algoritmo es utilizado para calcular el flujo óptico entre 2 imágenes que pertenecen a una secuencia. Para ejecturalo, es necesario importar las siguientes librerías:

```python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy as sp
import math
from scipy.interpolate import RegularGridInterpolator
import time

##Librerías customizadas
from hermite import  dht2
from HermiteRotado import principal_HR
from procesoResolucion import resolutionProcess
from warping import of_warp_multi, of_warp_multi_mod
from calculoColor import computeColor
```

Como se puede observar, hay librerías generadas justo para esta aplicación. Estas librerías son:

* **hermite:** esta librería tiene la implementación de la transformada de Hermite en imágenes. Esta implementación fue desarrollada por el Dr. Silván y utiliza a la función binomial como aproximación de la función gaussiana.
* **HermiteRotado:** incluye la implementación de la transformada de Hermite en imágenes hecha por el Dr. Leiner Barba. Esta implementación utiliza la discretización de la gaussiana. Además, incluye la etapa de rotación en la que se concentra mayor cantidad de energía en los coeficientes de los primeros órdenes.
* **procesoResolucion:** calcula las velocidades de flujo óptico a partir de los coeficientes de Hermite.
* **warping:** Interpolación necesaria para obtener el flujo óptico.

El flujo se calcula con la función **of_warp_multi** a partir de dos imágenes y los parámetros siguientes:

```python
## Hiperparámetros para el cálculo del flujo óptico
alpha=10; #100 sua  500   90    para el ground true en ultrasonido 10  7 % peso del funcional
gamma=1;   #%int. cte. 600  10   para el ground true en ultrasonido 1	 % peso del funcional
num_levels=1; #%40               =1  ;optimos para sinteticas3 %escalas
itera_outer=5;#%20               =3  ;optimos para sinteticas3 %iteraciones del programa, minimo 2
itera_iner=5;#%#50               =30 ;funciÃ³n de error, te permite corregir ciertas cosas en los resultdos como indeterminaciones
omega =1.8
truerange = 3;   
range1 = truerange * 1.04;
rotacion = 1 #Hermite clásico = 0, Hermite rotado = 1
#Cálculo del flujo óptico
u, v = of_warp_multi(img1, img2, alpha, gamma, num_levels, itera_outer, itera_iner, rotacion)
```

Las variables **u** y **v** son la velocidades del flujo óptico.

## Flujo paralelizado

Con el objetivo de acelerar el cálculo del flujo óptico contemplando que se analizarán secuencias de imágenes con diferente número de frames, se plantea paralelizar el proceso agregando la librería **joblib**:

```python
from joblib import Parallel, delayed
```

Las funciones **Parallel** y **delayed** de **joblib** son aplicadas sobre la función **of_warp_multi** de la siguiente manera:

```python
results = Parallel(n_jobs=mp.cpu_count(), prefer='threads')(delayed(of_warp_multi_mod)(volumen[idv-volumen.shape[0],:,:], volumen[idv+1-volumen.shape[0],:,:], alpha, gamma, num_levels, itera_oter[0], itera_iner[0], rotacion,sg[0]) for idv in range(volumen.shape[0]))
```

Esta implementación está en el código **prueba_paralelizacion.py** en dónde se calcula el flujo óptico de una secuencia de ultrasonido cardiaco.

Por último, el código **FlujoOptico_CAMUS_paralelo.py** sirve para calcular los flujos de las secuencias de la base de datos pública CAMUS.

    


