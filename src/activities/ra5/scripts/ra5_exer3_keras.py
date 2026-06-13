import os
os.environ['KERAS_BACKEND'] = 'tensorflow'

import keras
from keras import layers

# Creamos un modelo secuencial
model = keras.Sequential([
  # 1. Capa de entrada (Redimensiona una matriz de 28x28 a un vector plano de 784)
  layers.Flatten(input_shape=(28, 28)),
  
  # 2. Capa oculta densa (Totalmente conectada)
  layers.Dense(128, activation='relu'),
  
  # 3. Capa de salida (10 neuronas, una para cada clase posible)
  layers.Dense(10, activation='softmax')
])