import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import pickle

# Cargando los datos
data = pd.read_csv('time_series_covid_19_deaths.csv')

# Suponiendo que deseas trazar la serie de tiempo para la primera fila (Latitud)
x = range(len(data.columns) - 2)  # Ignorando las dos primeras columnas (Latitud y Longitud)
y = data.iloc[0, 2:].values  # Ignorando las dos primeras columnas (Latitud y Longitud)

# Trazando la serie de tiempo
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title('Serie de tiempo de casos de muertes por COVID-19')
plt.xlabel('Días desde 1/22/20')
plt.ylabel('Casos confirmados')
plt.grid(True)
plt.show()

# Creando y entrenando el modelo ARIMA con p=5 (datos anteriores) y d=0 (no diferencia)
model = ARIMA(y, order=(5, 0, 0))
model_fit = model.fit()

# Realizando predicciones
predictions = model_fit.forecast(steps=1)[0]

# Guardando el modelo entrenado utilizando pickle
filename = 'modelo_autoregresivo.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model_fit, file)

print("Predicción para el valor más reciente:", predictions)
