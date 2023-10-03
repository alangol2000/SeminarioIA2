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
data = pd.read_csv('concentlite.csv')
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values.reshape(-1, 1)

# Normalización de datos (escalar entre 0 y 1)
X = X / np.amax(X, axis=0)
y = y / 100

# Obtener la arquitectura de la red desde la entrada del usuario
num_layers = int(input("Numero de capas en la red: "))
layer_neurons = []
for i in range(num_layers):
    neurons = int(input(f"Numero de neuronas para la capa {i + 1}: "))
    layer_neurons.append(neurons)

# Definición de la arquitectura de la red neuronal
input_neurons = X.shape[1]
output_neurons = 1

# Inicialización aleatoria de los pesos de la red
weights = []
momentum_weights = []
for i in range(num_layers):
    if i == 0:
        weights.append(np.random.uniform(size=(input_neurons, layer_neurons[i])))
    else:
        weights.append(np.random.uniform(size=(layer_neurons[i - 1], layer_neurons[i])))
    momentum_weights.append(np.zeros_like(weights[i]))
weights.append(np.random.uniform(size=(layer_neurons[-1], output_neurons)))
momentum_weights.append(np.zeros_like(weights[-1]))

# Hiperparámetros
learning_rate = 0.1
momentum = 0.9  # Tasa de momentum
epochs = 10000

# Entrenamiento de la red neuronal con momentum
for epoch in range(epochs):
    # Propagacion
    layer_outputs = []
    layer_inputs = []
    for i in range(num_layers + 1):
        if i == 0:
            layer_input = np.dot(X, weights[i])
        else:
            layer_input = np.dot(layer_outputs[i - 1], weights[i])
        layer_inputs.append(layer_input)
        layer_output = sigmoid(layer_input)
        layer_outputs.append(layer_output)
    
    # Cálculo del error
    error = y - layer_outputs[-1]
    
    # Retropropagacion con momentum
    deltas = []
    for i in reversed(range(num_layers + 1)):
        if i == num_layers:
            delta = error * sigmoid_derivative(layer_outputs[i])
        else:
            delta = deltas[-1].dot(weights[i + 1].T) * sigmoid_derivative(layer_outputs[i])
        deltas.append(delta)
    
    deltas.reverse()
    
    for i in range(num_layers + 1):
        momentum_weights[i] = momentum * momentum_weights[i] + layer_outputs[i - 1].T.dot(deltas[i]) * learning_rate
        weights[i] += momentum_weights[i]

# Clasificación de los datos
layer_input = np.dot(X, weights[0])
layer_output = sigmoid(layer_input)
for i in range(1, num_layers + 1):
    layer_input = np.dot(layer_output, weights[i])
    layer_output = sigmoid(layer_input)
predicted_output = layer_output

# Visualización de la distribución de clases
plt.scatter(X[:, 0], X[:, 1], c=y.reshape(-1), cmap='coolwarm')
plt.title('Distribución de Clases para el Dataset concentlite')
plt.colorbar(label='Clase')
plt.show()
