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

#####-------------------------------------- Con los datos importados --------------------------------########

data = pd.read_csv('Data_Brasil_23_nov_2021.csv', sep=',')
data = data['total_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'Total Cases']
data

####### ---------------------- Definimos la función a ajustar -----------------------------#################

xdata = data.iloc[:,0:1]
ydata = data.iloc[:,1:2]

#print(ydata)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(xdata,ydata)

#############------------ visualising the linear regression model -----------------####################

plt.scatter(xdata,ydata, color='red')
plt.grid(True)
plt.plot(xdata, lin_reg.predict(xdata),color='blue')
plt.title("Linear model for total cases infections of COVID-19 in Brazil")
plt.xlabel('Time')
plt.ylabel('Infections')
plt.show()

# polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
X_poly = poly_reg.fit_transform(xdata)

X_poly     # prints X_poly

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,ydata)

# visualising polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(xdata)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,ydata)

X_grid = np.arange(xdata.min()[0], xdata.max()[0],0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(xdata,ydata, color='red')

plt.plot(X_grid, lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')

plt.title("Polynomial regression model for total cases infections of COVID-19 in Brazil")
plt.xlabel('Time')
plt.ylabel('Infections')
plt.grid(True)
plt.show()

fit_y = lin_reg2.predict(poly_reg.fit_transform(xdata))

fit_yy = []

for s in fit_y:
  fit_yy.append(s[0])

fit_y = np.array(fit_yy)

################ ------- Gráfica No.1 de los errores del modelo de crecimiento logístico ---------- #####################

ydata = ydata.to_numpy()

ydata_y = []

for s in ydata:
  ydata_y.append(s[0])

ydata = np.array(ydata_y)


#print(ydata)

delta = ydata - fit_y


n = 638 # Número total de datos.

Error_1 = np.abs(delta)/n ### Función de error 1 normalizada respecto al número total de datos

plt.plot(xdata, Error_1)
#plt.xlim([0, 420])
#plt.ylim([0,2])
plt.grid(True)
plt.title('Function Error 1 of polynomial regression model of Brazil COVID-19')
plt.legend(['Function of error 1 of polynomial regression Model'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Function of Error 1')
plt.show()



################ ------- Gráfica No.2 de los errores del modelo de Regresión Polinomial de ML ---------- #####################

Error_2 = ((ydata - fit_y)*(ydata - fit_y))/n ### Función de error 2 normalizada respecto al número total de datos

#print(Error_1)

plt.plot(xdata, Error_2)
#plt.xlim([0, 420])
#plt.ylim([0,2])
plt.grid(True)
plt.title('Function Error 2 of Polynomial Regression model of Brazil COVID-19')
plt.legend(['Function of error 2 of Polynomial Regression'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Function of Error 2')
plt.show()

################ ------- Gráfica No.3 de los errores del modelo de crecimiento logístico ---------- #####################

Error_3 = (ydata - fit_y)/ydata ### Función de error 3 normalizada respecto al valor de los datos reales

#print(Error_1)

plt.plot(xdata, Error_3)
#plt.xlim([0, 420])
#plt.ylim([-25,25])
plt.grid(True)
plt.title('Function Error 3 of Polynomial Regression model of Brazil COVID-19')
plt.legend(['Function of error 3 of Polynomial Regression Model'], loc="upper left")
plt.xlabel('Time')
plt.ylabel('Function of Error 3')
plt.show()
