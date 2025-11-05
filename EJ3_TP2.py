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
#%% 3b)
data = pd.DataFrame(columns=['Profundidad','Precisión'])
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

#%% 3c)
data1 = pd.DataFrame(columns=['Profundidad','Criterio','min_samples_split','min_samples_leaf','Precisión'])
for i in range(1,11,3):
    for j in range(2,20,5):
        for k in range(1,11,3):
            for elem in ['gini','entropy']:
                arbol = DecisionTreeClassifier(criterion=elem,
                splitter='best',
                max_depth=i,
                min_samples_split=j,
                min_samples_leaf=k,
                max_features=None,
                random_state=2)

                scores = cross_val_score(arbol, X_train, y_train, cv=5)
                data1.loc[len(data1)]=[i,elem,j,k,scores.mean()] 

