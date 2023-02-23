import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib


"""
DATA:
    CRIM: Tasa de criminalidad per cápita por ciudad.
    ZN: Proporción de terrenos residenciales zonificados para lotes de más de 25,000 pies cuadrados.
    INDUS: Proporción de acres de negocios no minoristas por ciudad.
    CHAS: Variable ficticia de Charles River (1 si el tramo limita con el río; 0 en caso contrario).
    NOX: Concentración de óxidos nítricos (partes por 10 millones).
    RM: Número medio de habitaciones por vivienda.
    AGE: Proporción de unidades ocupadas por sus propietarios construidas antes de 1940.
    DIS: Distancias ponderadas a cinco centros de empleo en Boston.
    RAD: Índice de accesibilidad a carreteras radiales.
    TAX: Tasa de impuesto sobre la propiedad de valor total por cada $10,000.
    PTRATIO: Proporción alumno-profesor por ciudad.
    B: 1000 (Bk - 0.63)^2 donde Bk es la proporción de personas negras por ciudad.
    LSTAT: Porcentaje de población de estatus bajo.
    
TARGET:
    MEDV: Precios medios de las viviendas ocupadas por sus propietarios en miles de dólares.
"""

# Especificar el directorio donde se almacenarán los datos.
data_dir = os.path.join(os.getcwd(), "data")

# Descargar el dataset de Boston House Prices en el directorio especificado.
boston = fetch_openml(name='boston', version=1, data_home=data_dir)

data = pd.DataFrame(boston.data, columns=boston.feature_names)  # DataFrame con todos los datos de viviendas.

# Estandarizar los datos usando StandardScaler.
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Normalizar los datos usando MinMaxScaler.
normalizer = MinMaxScaler()
data_normalized = normalizer.fit_transform(data)

# Dividir los datos en conjuntos de entrenamiento y de prueba.
x_train, x_test, y_train, y_test = train_test_split(data_normalized, boston.target, test_size=0.2, random_state=42)

# Crear un modelo de regresión lineal y ajustarlo a los datos de entrenamiento.
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Hacer predicciones en los datos de prueba y calcular el error cuadrático medio.
y_prediction = regressor.predict(x_test)
mse = mean_squared_error(y_test, y_prediction)

# Hacer predicciones en nuevos datos.
new_data = [[0.0632, 10.0, 2.31, 0, 0.538, 6.575, 61.2, 4.0900, 1.99, 298.0, 15.3, 354.90, 5.98]]
new_data_normalized = scaler.transform(new_data)
prediction = regressor.predict(new_data_normalized)

# Guardar el modelo en un archivo
joblib.dump(regressor, 'models/modelo_entrenado.pkl')

# Cargar el modelo desde el archivo
loaded_model = joblib.load('models/modelo_entrenado.pkl')

# Hacer predicciones en nuevos datos utilizando el modelo cargado
predictions = loaded_model.predict(new_data_normalized)



# ----------------------------------------    P R I N T S    ---------------------------------------------------
print(' ')
print(' ')

op = None
while op != '':
    op = input('Presione Enter...')

print(' ')
print("Error cuadrático medio:", mse)
print(' ')
# Imprimir la predicción
print("Predicción:", predictions)
print(' ')

# # Mostrar las características y las etiquetas del dataset
# print(' ')
print(' ')
# print('DATA')
# print(type(data))
print(data)
print(' ')
print(boston.target)
print(' ')
# print(' ')
# print('data_scaled')
# print(type(data_scaled))
# print(data_scaled)
# print(' ')
# print(' ')
# print('data_normalized')
# print(type(data_normalized))
# print(data_normalized)
# print(' ')
# print(' ')
# --------------------------------------------------------------------------------------------------------------


