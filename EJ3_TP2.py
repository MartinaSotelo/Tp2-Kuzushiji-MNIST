from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import duckdb as dd
import numpy as np
from sklearn.model_selection import cross_val_score

carpeta = "C:/Users/perei/Downloads/Datos_para_el_TP2/"

kminst = pd.read_csv(carpeta+"kmnist_classmap_char.csv")
kuzu_full = pd.read_csv(carpeta+"kuzushiji_full.csv")

#%%

X = kuzu_full.iloc[:,:784]
y = kuzu_full['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

data = pd.DataFrame(columns=['profundidad','presicion'])
for i in range(1,10,2):
    arbol = DecisionTreeClassifier(criterion='gini',
        splitter='best',
        max_depth=i,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=None)

    scores = cross_val_score(arbol, X_train, y_train, cv=5)
    data.loc[len(data)]=[i,scores.mean()]
# arbol.fit(X_train, y_train) # Entrenamiento del modelo
# prediction = arbol.predict(X_test) # Generamos las predicciones // llamamos al modelo
# #print(prediction)
# accuracy = accuracy_score(y_test, prediction)

print("Precisión en cada fold:", scores)
print("Precisión promedio:", scores.mean())

