from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

carpeta = ""

kminst = pd.read_csv(carpeta+"kmnist_classmap_char.csv")
kuzu_full = pd.read_csv(carpeta+"kuzushiji_full.csv")

#%%

X = kuzu_full.iloc[:,:784]
y = kuzu_full['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#%% 3b)
data = pd.DataFrame(columns=['profundidad','presicion'])
for i in range(1,11,3):
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

# Luego de realizar varias pruebas, se llega a la conclusión de que la presicion máxima es de aproximadamente el 73%
# con max_depth = 30, el accuracy es del 0.733%, mientras que en max_depth = 38, era del 0.735%
# Y mas o menos a partir de profundidad 40, es que el accuracy deja de subir
#%% 3c)
# !!Este codigo tarda mucho en correr!! 

lista = ['gini','entropy']
data1 = pd.DataFrame(columns=['profundidad','criterio','min_samples_split','min_samples_leaf','presicion'])
for i in range(1,11,3):
    for j in range(2,20,5):
        for k in range(1,11,3):
            for elem in lista:
                arbol = DecisionTreeClassifier(criterion=elem,
                splitter='best',
                max_depth=i,
                min_samples_split=j,
                min_samples_leaf=k,
                max_features=None,
                random_state=2)

                scores = cross_val_score(arbol, X_train, y_train, cv=5)
                data1.loc[len(data1)]=[i,elem,j,k,scores.mean()]  
                

data2 = pd.DataFrame(columns=['Profundidad','Criterio','min_samples_split','min_samples_leaf','Precisión'])
for i in range(2,50,10):
    for j in range(2,50,10):
        arbol = DecisionTreeClassifier(
        criterion='entropy',
        splitter='best',
        max_depth=10,
        min_samples_split=j,
        min_samples_leaf=i,
        random_state=2)
        scores = cross_val_score(arbol, X_train, y_train, cv=5)
        data2.loc[len(data2)]=[10,'entropy',j,i,scores.mean()] 

#%%

graf = plt.scatter(data2['min_samples_split'],data2['min_samples_leaf'], c=data2['Precisión'])
plt.xlabel('min_samples_split')
plt.ylabel('min_samples_leaf')
plt.colorbar()
plt.show()

cbar = plt.colorbar(graf)
cbar.set_label('Precisión')
#%% 3d)
arbol = DecisionTreeClassifier(criterion='entropy',
        splitter='best',
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=0)

scores = cross_val_score(arbol, X_train, y_train, cv=5)
desempeño = str(scores.mean()*100)
print("Desempeño del modelo de árbol final: " + desempeño + "%" )

#%%
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))

