import pandas as pd
import duckdb as dd
import re
import numpy as np
import sklearn

carpeta = ""

kminst = pd.read_csv(carpeta+"kmnist_classmap_char.csv")
kuzu_full = pd.read_csv(carpeta+"kuzushiji_full.csv")

#%%

import pandas as pd
import matplotlib.pyplot as plt

X = píxeles (todas menos la última columna)
X = kuzu_full.iloc[:, :-1].values

y = etiquetas (última columna)
y = kuzu_full.iloc[:, -1].values

from sklearn.model_selection import train_test_split

Dividimos los datos
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_imgs = X_train.reshape(-1, 28, 28)

Mostramos las primeras 9 imágenes
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_imgs[i], cmap='gray')
    ax.set_title(f"Etiqueta: {y_train[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()

#%%
import matplotlib.pyplot as plt
consulta = """
           SELECT *
           FROM kuzu_full
           WHERE label = 4 OR label = 5
           """
kuzu_full_4_5 = dd.query(consulta).df()

X = kuzu_full_4_5.iloc[:, :-1].values

y = etiquetas (última columna)
y = kuzu_full_4_5.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_imgs = X_train.reshape(-1, 28, 28)

Mostramos las primeras 9 imágenes
fig, axes = plt.subplots(3, 3, figsize=(6, 6))

for i, ax in enumerate(axes.flat):
    ax.imshow(X_train_imgs[i], cmap='gray')
    ax.set_title(f"Etiqueta: {y_train[i]}")
    ax.axis('off')

plt.tight_layout()
plt.show()
#%%

consulta = """
           SELECT COUNT(label)
           FROM kuzu_full_4_5
           GROUP BY label
           """
a = dd.query(consulta).df()
