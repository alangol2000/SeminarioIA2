
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar dataset
# Para el punto 1 utilizamos el archivo "spheres1d10.csv"
data = pd.read_csv("spheres1d10.csv")
# Para el punto 2 utilizamos el nuevo dataset generado modificando la tabla señalada en el archivo de Practica 1
# data = pd.read_csv("nuevoDataset.csv")

# Dividir los datos en características y destino.
X = data.iloc[:, :-1]  # características
y = data.iloc[:, -1]   # destino

# Modelo de perceptron simple
model = Perceptron()

for i in range(5):
    # Dividir los datos en conjuntos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    accuracy = model.score(X_test, y_test)
    print("Model Accuracy for partition ", i, ": ", accuracy)

accuracies = []
for i in range(5):
    # Dividir los datos en conjuntos de entrenamiento y prueba.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    # Entrenar el modelo
    model.fit(X_train, y_train)

    # Evaluar el modelo
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)
    print("Model Accuracy for partition ", i, ": ", accuracy)

#Graficas

plt.figure(figsize=(10, 5))
plt.plot(range(5), accuracies, marker='o', linestyle='--')
plt.title('Model Accuracy for Each Partition')
plt.xlabel('Partition Number')
plt.ylabel('Accuracy')
plt.grid()
plt.show()

y_pred = model.predict(X_test)

plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()