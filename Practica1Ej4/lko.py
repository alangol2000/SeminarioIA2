import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
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

# Crear el modelo MLP (Perceptrón Multicapa)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Generar un número aleatorio entre 1 y 10 para k
k = np.random.randint(1, 11)

# Utilizar el método de validación cruzada Leave-k-Out para evaluar el modelo
scores = cross_val_score(mlp, X_scaled, y, cv=k)

# Imprimir el error de clasificación, la puntuación promedio y la desviación estándar
error_clasificacion = 1 - scores.mean()
puntuacion_promedio = scores.mean()
desviacion_estandar = scores.std()

print("Resultados de Validación Cruzada Leave-{}-Out:".format(k))
print("Error de Clasificación: {:.2f}".format(error_clasificacion))
print("Puntuación Promedio: {:.2f}".format(puntuacion_promedio))
print("Desviación Estándar: {:.2f}".format(desviacion_estandar))

# Reducir la dimensionalidad a 2D para visualización utilizando PCA
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
