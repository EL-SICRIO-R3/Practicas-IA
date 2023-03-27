import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

#inicializar datos de entrenamiento
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

#creacion de la red neuronal
model = Sequential()
model.add(Dense(250, input_dim=1, activation='relu'))
model.add(Dense(250, input_dim=1, activation='relu'))
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(1, activation='linear'))

#complar la red
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

#entrenar la red
model.fit(celsius, fahrenheit, epochs=1500)

#guardar los pesos en un archivo H5
model.save('celsius_a_fahrenheit.h5')

#cargar éste archivo para su prueba
model = load_model('celsius_a_fahrenheit.h5')

#cargar los datos con los que se testeará la red
test_data = np.array([-30, -5, 10, 120, 30], dtype=float)

#ejecutar la red
predictions = model.predict(test_data)

#mostrar los resultaos para cada dato de prueba ingresado
for i in range(len(test_data)):
    print(f'{test_data[i]} grados Celsius = {predictions[i][0]} grados Fahrenheit') 