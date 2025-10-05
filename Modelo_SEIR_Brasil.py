###### Paquetes y librerias a importar ######################
 
import numpy as np
import math
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

############# Parámetros del artículo OPTIMAL VACCINATION STRATEGIES FOR AN SEIR MODEL OF INFECTIOUS DISEASES WITH LOGISTIC GROWTH ########

a = 0.1 # disease induced death rate
b = 0.525 # natural birth rate
c = 0.001 # incidence coefficient
d = 0.5 # natural death rate
e = 0.5 # exposed to infectious rate
g = 0.1 # natural recovery rate
u = 0.95 

################# Definimos las ecuaciones diferenciales acopladas a resolver #####################
 
def equations_systems_0(equations,t):
    S, E, I, R, N, U, W = equations
    dS_t = b*N - d*S - c*S*I - U*S
    dE_t = c*S*I - (e+d)*E
    dI_t = e*E - (g+a+d)*I
    dR_t = g*I - d*R + U*S
    dN_t = (b-d)*N - a*I
    N = S + E + I + R
    dU_t = u*math.exp(-e*t)
    dW_t = U*S
    return dS_t, dE_t, dI_t, dR_t, dN_t, dU_t, dW_t

S0 = 1000
E0 = 100
I0 = 50
R0 = 15
N0 = 1165
U0 = 0.02
W0 = 0

initial_values = S0, E0, I0, R0, N0, U0, W0
 
t = np.linspace(0,500,num=3000)
 
sol=odeint(equations_systems_0, initial_values, t)
 
plt.plot(t, sol[:, 0], label="S_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()

plt.plot(t, sol[:, 1], label="E_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()

plt.plot(t, sol[:, 2], label="I_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()

plt.plot(t, sol[:, 3], label="R_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()

plt.plot(t, sol[:, 4], label="N_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()

plt.plot(t, sol[:, 5], label="U_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()

plt.plot(t, sol[:, 6], label="W_t")
plt.xlim([0, 300])
plt.grid("True")
plt.legend()
plt.show()
