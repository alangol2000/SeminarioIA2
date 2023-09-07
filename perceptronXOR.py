import numpy as np
import matplotlib.pyplot as plt


def step_function(x):
    return 1 if x >= 0 else -1


def train_perceptron(X, y, learning_rate, max_epochs):
    num_inputs = X.shape[1]
    num_samples = X.shape[0]
    

    weights = np.random.rand(num_inputs)
    bias = np.random.rand()
    
    for epoch in range(max_epochs):
        error = 0
        for i in range(num_samples):
            net_input = np.dot(X[i], weights) + bias
            output = step_function(net_input)
            delta = y[i] - output
            weights += learning_rate * delta * X[i]
            bias += learning_rate * delta
            
            error += delta ** 2
        
        error /= 2
        if error == 0:
            break
    
    return weights, bias
def test_perceptron(X, weights, bias):
    num_samples = X.shape[0]
    predictions = []
    
    for i in range(num_samples):
        net_input = np.dot(X[i], weights) + bias
        output = step_function(net_input)
        predictions.append(output)
    
    return predictions
X_train = np.genfromtxt('XOR_trn.csv', delimiter=',')
y_train = np.genfromtxt('XOR_trn.csv', delimiter=',', usecols=-1)

X_test = np.genfromtxt('XOR_tst.csv', delimiter=',')
y_test = np.genfromtxt('XOR_tst.csv', delimiter=',', usecols=-1)
learning_rate = 1
max_epochs = 4
weights, bias = train_perceptron(X_train, y_train, learning_rate, max_epochs)
predictions = test_perceptron(X_test, weights, bias)

print("Predicciones en el conjunto de prueba:")
for i, prediction in enumerate(predictions):
    print(f"Patrón {i+1}: {prediction}")
plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], marker='o', label='Clase 1', c='b')
plt.scatter(X_train[y_train == -1][:, 0], X_train[y_train == -1][:, 1], marker='x', label='Clase -1', c='r')
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = np.dot(np.c_[xx.ravel(), yy.ravel()], weights[:2]) + bias
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, levels=[0], linewidths4=2, colors='g')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend(loc='upper right')
plt.title('Separación de Clases')
plt.show()