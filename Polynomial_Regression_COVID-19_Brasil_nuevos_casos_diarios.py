###### Paquetes y librerias a importar ######################
import numpy as np
import scipy as sp
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd
from pylab import rcParams

###### importamos los datos con los que vamos a trabajar #######
data = pd.read_csv('data_Colombia (1).csv', sep=',')
data = data['new_cases']
data = data.reset_index(drop=False)
data.columns = ['Timestep', 'New Cases']
print(data)

####### Gráfica de los nuevos casos de contagio en Colombia ######
xdata = data.iloc[:,0:1].values
ydata = data.iloc[:,1:2].values
rcParams['figure.figsize'] = 14, 8
SMALL_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.plot(xdata, ydata, color='blue', linewidth=3)
#plt.title("New cases of infections of COVID-19 in Colombia")
plt.xlabel('Time [days]')
plt.ylabel('Number of new Infections')
plt.grid(True)
plt.show()

#### Ajuste de la gráfica usando la regresión lineal #####
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(xdata,ydata)

#############------------ visualising the linear regression model -----------------##
rcParams['figure.figsize'] = 14, 8
SMALL_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.grid(True)
plt.plot(xdata,ydata, color='blue', linewidth=3, label="Daily cases")
plt.plot(xdata, lin_reg.predict(xdata),color="red", linewidth=3, label="Linear_regresion")
plt.title("Linear model for total cases infections of COVID-19 in Colombia")
plt.xlabel('Time')
plt.ylabel('Infections')
plt.legend()
plt.show()


#### Ajuste de la gráfica usando la regresión polinomial #####
# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 8)
X_poly = poly.fit_transform(xdata)
poly.fit(X_poly, ydata)
lin2 = LinearRegression()
lin2.fit(X_poly, ydata)


# Visualising the Polynomial Regression results
rcParams['figure.figsize'] = 14, 8
SMALL_SIZE = 20
plt.rc('font', size=SMALL_SIZE)
plt.rc('axes', titlesize=SMALL_SIZE)
plt.plot(xdata, ydata, color = 'blue', linewidth=3, label="Daily cases")
plt.plot(xdata, lin2.predict(poly.fit_transform(xdata)), color = 'red', linewidth=3, label="polynomial_regresion")
plt.title('Polynomial Regression')
plt.xlabel('Temperature')
plt.ylabel('Pressure')
plt.legend()
plt.grid(True)
plt.show()
