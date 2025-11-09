
"""
Grupo Import_milanesas
Integrantes: 
Dulio Joaquin, 
Risuleo Franco, 
Perez Sotelo Martina

Entrenamos modelo knn para clasificacion binaria
de las clases 4 y 5.

"""

import pandas as pd
import duckdb as dd

carpeta = "/home/martina/import_milanesas_TP02/"

kminst = pd.read_csv(carpeta+"kmnist_classmap_char.csv")
kuzu_full = pd.read_csv(carpeta+"kuzushiji_full.csv")

#%%

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# #X = píxeles (todas menos la última columna)
# X = kuzu_full.iloc[:, :-1].values

# #y = etiquetas (última columna)
# y = kuzu_full.iloc[:, -1].values

# #Dividimos los datos
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )

# X_train_imgs = X_train.reshape(-1, 28, 28)

# #Mostramos las primeras 9 imágenes
# fig, axes = plt.subplots(3, 3, figsize=(6, 6))

# for i, ax in enumerate(axes.flat):
#     ax.imshow(X_train_imgs[i], cmap='gray')
#     ax.set_title(f"Etiqueta: {y_train[i]}")
#     ax.axis('off')

# plt.tight_layout()
# plt.show()

#%%
consulta = """
           SELECT *
           FROM kuzu_full
           WHERE label = 4 OR label = 5
           """
kuzu_full_4_5 = dd.query(consulta).df()

X = kuzu_full_4_5.iloc[:, :-1].values

#y = etiquetas (última columna)
y = kuzu_full_4_5.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_imgs = X_train.reshape(-1, 28, 28)

#Mostramos las primeras 9 imágenes
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_imgs[i], cmap='gray')
    ax.set_title(f"Etiqueta: {y_train[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
#%%

#Con esta consulta vemos que estan balanceados los datos. Hay 70000 de cada clase.

consulta = """
           SELECT COUNT(label)
           FROM kuzu_full_4_5
           GROUP BY label
           """
balance = dd.query(consulta).df()
#%%
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import random


#armar un conjunto de pixeles que contenga a estos. Ver que pasa. Yo creo que tendria que dar mejor la prediccion...
"""los pixeles con mayor variabilidad entre clase 4 y clase 5 son:
122:111.02075871986933
123:111.9380869547499
150:111.61850063989782
151:112.16713747463827
178:111.71238033694596
179:111.39979353322525
331:111.5050539917653
342:111.50445185067507
343:111.3912744033587
359:111.74018106972679
360:113.12511998004403
370:112.56159549792135
371:111.91295842000963
388:113.56507449800914
389:111.72072661315468
398:112.29384775491697
399:111.07586680489429
416:112.29931665864497
417:112.63029018562779
426:111.66827645042022
445:111.12800850129747"""


#Vemos que los bordes suelen dar peor que los atributos más centrados
modelos_knnC = pd.DataFrame(columns=["Cantidad_atributos", "Atributos_usados", "Accuracy"])
#Bordes-bordes-medio-seguidos-cualquiercosa-atributos con mayor variabilidad.
subconjuntos_de_atributos4 = [[0,27,28,55],[56,83,84,168],[17,62,92,120],[444,445,446,447],[578,612,499,702],[416,370,360,388]]
subconjuntos_de_atributos5 = [[0,27,28,55,56],[56,83,84,168,783],[17,62,92,120,355],[444,445,446,447,448],[578,612,499,702,550],[416,151,370,360,388]]
subconjuntos_de_atributos7 = [[0,27,28,55,56,83,84],[56,83,84,168,196,224,252],[17,62,92,120,264,148,404],[444,445,446,447,448,449,450],[578,612,499,702,123,389,12],[151,388,398,416,370,360,388]]
s = [subconjuntos_de_atributos4, subconjuntos_de_atributos5, subconjuntos_de_atributos7]
for grupo_subconjuntos in s:
    cant_atributos = len(grupo_subconjuntos[0])
    for atributos in grupo_subconjuntos:
        sub_X_train = X_train[:, atributos]
        sub_X_test = X_test[:, atributos]
        knn = KNeighborsClassifier()
        knn.fit(sub_X_train, y_train)
        y_pred = knn.predict(sub_X_test)
        accuracy = accuracy_score(y_test, y_pred)
        modelos_knnC.loc[len(modelos_knnC)] = [cant_atributos, atributos, accuracy]



#%%
#Este código genera atributos aleatorios, entrena un knn en base a esos atributos (testea el mejor n
#en enes y usa ese), y guarda cómo le fue a cada modelo (accuracy) con sus atributos elegidos, junto con la cantidad
#de atributos que usó y el mejor n. A lo mejor en vez de testear el mejor n, habría que guardar todos y encontrar
#alguna relación entre cantidad de atributos y n usado. Algo como que a mayor cantidad de atributos, peores
#terminan siendo los n chicos?

n_atributos = [3,4,5,8,10,15,20]
modelos_knn = pd.DataFrame(columns=["Cantidad_atributos", "Atributos_usados", "Mejor_n", "Accuracy"])
enes = [4,10,20] #Habría que ver qué números de vecinos probar. Puse esos de prueba nomas
for cant_atributos in n_atributos:
    subconjuntos_de_atributos = []
    for i in range(6):
        subconjunto = random.sample(range(0, 784), cant_atributos)
        subconjuntos_de_atributos.append(subconjunto)
    for atributos in subconjuntos_de_atributos:
        mejor_accuracy = 0
        mejor_n = enes[0]
        sub_X_train = X_train[:, atributos]
        sub_X_test = X_test[:, atributos]
        for n in range(2,30,5):
            knn = KNeighborsClassifier(n_neighbors=n)
            knn.fit(sub_X_train, y_train)
            y_pred = knn.predict(sub_X_test)
            accuracy = accuracy_score(y_test, y_pred)
            if accuracy > mejor_accuracy:
                mejor_accuracy = accuracy
                mejor_n = n
        modelos_knn.loc[len(modelos_knn)] = [cant_atributos, atributos, mejor_n, mejor_accuracy]




#Cómo me dijo chatgpt que lo hiciera con cross validation. Capaz sirve para el 3
#from sklearn.model_selection import cross_val_score, StratifiedKFold

# modelos_knn_cross_validation = pd.DataFrame(columns=["Atributos_usados", "Mejor_numero_neighbors", "Accuracy"])
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# for atributos in subconjuntos_de_atributos:
#     mejor_accuracy = 0
#     mejor_n = enes[0]
#     sub_X_train = X_train[:, atributos]
#     sub_X_test = X_test[:, atributos]
#     for n in enes:
#         knn = KNeighborsClassifier(n_neighbors=n)
#         scores = cross_val_score(knn, sub_X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
#         mean_accuracy = scores.mean()
#         if mean_accuracy > mejor_accuracy:
#             mejor_accuracy = mean_accuracy
#             mejor_n = n
#     knn_final = KNeighborsClassifier(n_neighbors=mejor_n)
#     knn_final.fit(sub_X_train, y_train)
#     test_acc = knn_final.score(sub_X_test, y_test)
#     modelos_knn_cross_validation.loc[len(modelos_knn_cross_validation)] = [atributos, mejor_n, mejor_accuracy]
    
#con cross validation o no dan bastante parecidos igual

#Usé accuracy, no veo por qué otra métrica sería más adecuada
# %%

#copie y pegue de gemini:

import matplotlib.pyplot as plt
import pandas as pd
# Asumo que modelos_knnC y modelos_knn ya están definidos y cargados

# --- Gráfico 1: Precisión vs. Cantidad de Atributos (Atributos Fijos y Aleatorios) ---

plt.figure(figsize=(10, 6))

# A. Atributos Fijos (modelos_knnC)
# Usamos un color distintivo para estos puntos estratégicamente elegidos
plt.scatter(
    modelos_knnC['Cantidad_atributos'],
    modelos_knnC['Accuracy'],
    color='red',
    marker='s', # Marcador cuadrado para diferenciar
    s=100,      # Tamaño del marcador
    label='Conjuntos Fijos (Ubicación Específica)'
)

# B. Atributos Aleatorios (modelos_knn)
# Usamos un color y marcador para los aleatorios
plt.scatter(
    modelos_knn['Cantidad_atributos'],
    modelos_knn['Accuracy'],
    color='blue',
    alpha=0.6,
    label='Conjuntos Aleatorios'
)

# C. Configuración
plt.title('Precisión (Accuracy) vs. Cantidad de Atributos (Clases 4 y 5)')
plt.xlabel('Cantidad de Atributos (Píxeles) Usados')
plt.ylabel('Precisión (Accuracy)')
plt.ylim(0.5, 1.0) # Ajusta el límite si es necesario
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# --- Gráfico 2: Precisión vs. Mejor 'k' (Solo para Atributos Aleatorios) ---

plt.figure(figsize=(10, 6))

# Scatter plot que relaciona la precisión con el mejor 'k' encontrado para cada subconjunto
plt.scatter(
    modelos_knn['Mejor_n'],
    modelos_knn['Accuracy'],
    c=modelos_knn['Cantidad_atributos'], # Usamos la cantidad de atributos como color
    cmap='viridis',
    s=100
)

# C. Configuración
plt.title('Precisión (Accuracy) vs. Mejor Valor de k (Vecinos)')
plt.xlabel('Mejor Valor de k (Mejor n)')
plt.ylabel('Precisión (Accuracy)')
plt.ylim(0.5, 1.0)
cbar = plt.colorbar(label='Cantidad de Atributos') # Añade una barra de color
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
