
"""
Grupo Import_milanesas
Integrantes: 
Dulio Joaquin, 
Risuleo Franco, 
Perez Sotelo Martina

"""

import pandas as pd
import duckdb as dd

carpeta = "C:/Users/risul/OneDrive/Escritorio/TP2/"

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

#Vemos que los bordes suelen dar peor que los atributos más centrados
modelos_knnC = pd.DataFrame(columns=["Cantidad_atributos", "Atributos_usados", "Accuracy"])
#Bordes-bordes-medio-seguidos-cualquiercosa
subconjuntos_de_atributos4 = [[0,27,28,55],[56,83,84,168],[17,62,92,120],[444,445,446,447],[578,612,499,702]]
subconjuntos_de_atributos5 = [[0,27,28,55,56],[56,83,84,168,783],[17,62,92,120,355],[444,445,446,447,448],[578,612,499,702,550]]
subconjuntos_de_atributos7 = [[0,27,28,55,56,83,84],[56,83,84,168,196,224,252],[17,62,92,120,264,148,404],[444,445,446,447,448,449,450],[578,612,499,702,123,389,12]]
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
