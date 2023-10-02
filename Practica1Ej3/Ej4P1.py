import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Función de activación (sigmoid)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación
def sigmoid_derivative(x):
    return x * (1 - x)

# Lectura de datos desde el archivo CSV
data = pd.read_csv("concentlite.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Normalización de datos (escalar entre 0 y 1)
X = X / np.amax(X, axis=0)
y = y / 100

# Definición de la arquitectura de la red neuronal
input_neurons = X.shape[1]
hidden_neurons = 8  # Número de neuronas en la capa oculta
output_neurons = 1

# Inicialización aleatoria de los pesos de la red
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Hiperparámetros
learning_rate = 0.1
epochs = 10000

# Entrenamiento de la red neuronal
for epoch in range(epochs):
    # Capa de entrada a capa oculta
    input_hidden = np.dot(X, weights_input_hidden)
    output_hidden = sigmoid(input_hidden)
    
    # Capa oculta a capa de salida
    input_output = np.dot(output_hidden, weights_hidden_output)
    output = sigmoid(input_output)
    
    # Cálculo del error
    error = y - output
    
    # Retropropagación y ajuste de pesos
    d_output = error * sigmoid_derivative(output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(output_hidden)
    
    weights_hidden_output += output_hidden.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

# Clasificación de los datos
input_hidden = np.dot(X, weights_input_hidden)
output_hidden = sigmoid(input_hidden)
input_output = np.dot(output_hidden, weights_hidden_output)
predicted_output = sigmoid(input_output)

# Visualización de la distribución de clases
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='coolwarm')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Distribución de Clases para el Dataset concentlite')
plt.colorbar(label='Clase')
plt.show()