
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo 'irisbin.csv'
data = pd.read_csv('irisbin.csv', header=None)

# Extraer características (primeras 4 columnas) y etiquetas (últimas 3 columnas)
X = data.iloc[:, :4].values
y = data.iloc[:, 4:].values

# Convertir las etiquetas codificadas en one-hot a su equivalente ordinal
y = np.argmax(y, axis=1)

# Escalar las características para asegurar que estén en la misma escala
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Crear el modelo MLP (Perceptrón Multicapa)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Entrenar el modelo con los datos de entrenamiento
mlp.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
predictions = mlp.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, predictions)
print("Precisión del modelo: {:.2f}%".format(accuracy * 100))

# Reducir la dimensionalidad a 2D para la visualización utilizando PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Graficar la proyección 2D de las clases
plt.figure(figsize=(8, 6))
for i in range(3):
    indices = y == i
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f'Clase {i}')

plt.title('Proyección en dos dimensiones de la distribución de clases para el conjunto de datos Iris')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.legend()
plt.show()
