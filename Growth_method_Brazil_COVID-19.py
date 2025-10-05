import numpy as np
import pandas as pd
import seaborn as sns
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import torch
import math
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#def my_logistic(t):
#    return 10000 / (1 + 99999 * math.exp(-2 * t))

#pd.DataFrame({'Time':x, 'Infections':y})

#x = np.linspace(0, 20, 100)
#y = [my_logistic(i) for i in x]
#plt.grid(True)
#plt.plot(x, y)

#####-------------------------------------- Con los datos importados --------------------------------########

data = pd.read_csv('Data_Brasil_23_nov_2021.csv', sep=',')
data = data['total_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total Cases']
data    

####### ---------------------- Definimos la función a ajustar -----------------------------#################

xdata = np.array(data['Timestep']) + 1
ydata = np.array(data['Total Cases'])

from scipy.optimize import curve_fit

def f(x, a, b, c, d):
    y = a / (1. + np.exp(-c * (x - d))) + b
    return y

parameters, covariance = curve_fit(f, xdata, ydata)

fit_a = parameters[0]
fit_b = parameters[1]
fit_c = parameters[2]
fit_d = parameters[3]

print(fit_a, fit_b, fit_c, fit_d)

fit_y = f(xdata, fit_a, fit_b, fit_c, fit_d)

###################--------- Realizamos la gráfica del modelo matemático y los datos reales ----------------###########

plt.scatter(xdata,ydata,edgecolors='r',color='g', linewidth=2)
plt.plot(xdata, fit_y, '--b', linewidth=3.5)
#plt.xlim([0, 420])
plt.grid(True)
plt.title('Logistic model vs real data in Brazil COVID-19')
plt.legend(['Logistic Model', 'Real Data'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Infections')
plt.show()

################ ------- Gráfica No.1 de los errores del modelo de crecimiento logístico ---------- #####################

n = 637 # Número total de datos.

Error_1 = (ydata - fit_y)/n ### Función de error 1 normalizada respecto al número total de datos  

#print(Error_1)

plt.plot(xdata, Error_1)
#plt.xlim([0, 420])
#plt.ylim([0,2])
plt.grid(True)
plt.title('Function Error 1 of Logistic model of Brazil COVID-19')
#plt.legend(['Function of error 1 of Logistic Model'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Function of Error 1')
plt.show()

################ ------- Gráfica No.2 de los errores del modelo de crecimiento logístico ---------- #####################

Error_2 = ((ydata - fit_y)*(ydata - fit_y))/n ### Función de error 2 normalizada respecto al número total de datos  

#print(Error_1)

plt.plot(xdata, Error_2)
#plt.xlim([0, 420])
#plt.ylim([0,2])
plt.grid(True)
plt.title('Function Error 2 of Logistic model of Brazil COVID-19')
#plt.legend(['Function of error 2 of Logistic Model'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Function of Error 2')
plt.show()

################ ------- Gráfica No.3 de los errores del modelo de crecimiento logístico ---------- #####################

Error_3 = (ydata - fit_y)/ydata ### Función de error 3 normalizada respecto al valor de los datos reales  

#print(Error_1)

plt.plot(xdata, Error_3)
#plt.xlim([50, 420])
#plt.ylim([-2,2])
plt.grid(True)
plt.title('Function Error 3 of Logistic model of Brazil COVID-19')
#plt.legend(['Function of error 3 of Logistic Model'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Function of Error 3')
plt.show()
