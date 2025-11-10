
"""
Grupo Import_milanesas
Integrantes: 
Dulio Joaquin, 
Risuleo Franco, 
Perez Sotelo Martina

Este archivo contiene todo el codigo utilizado para realizar
el analisis exploratorio de los datos, los experimentos de clasificacion binaria
y multiclase.
Todos los graficos, tablas y resultados se encuentran aca detallados.
Separamos el codigo en tres secciones correspondientes al punto 1, 2 y 3 del TP.

"""

# importamos todas las librerias que vamos a utilizar
import pandas as pd
import numpy as np
import duckdb as dd
import random as rd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

#%%###################################################
#                   PUNTO 1 
######################################################
''' En esta seccion realizamos el Analisis exploratorio del dataset
todos los graficos correspondientes a la introduccion y seccion analisis exploratorio del informe
se encuentran aca'''

#%%

carpeta = "/home/martina/import_milanesas_TP02/" #poner nombre del path donde se encuentre los archivos


archivo1= pd.read_csv(carpeta+"kmnist_classmap_char.csv") #leemos los archivos
archivo2= pd.read_csv(carpeta+"kuzushiji_full.csv")

# %% Armo un grafico para visualizar una imagen por cada clase

#armo la figura
fig, axes = plt.subplots(2, 5, figsize=(26,10))
fig.suptitle("Grafico por Clase",y=1, fontsize=32)
axes = axes.flatten()
   
for i, clase in enumerate(range(10)):
    ax = axes[i] #selecciono el espacio del grafico
    
    #selecciono la clase
    consulta = f"""
               SELECT *
               FROM archivo2
               WHERE label = {clase}
               """
    clase_= dd.query(consulta).df()
    clase_.drop('label', axis=1, inplace=True) 
       
    #elijo un representante aleatorio de cada clase
    claseRep = clase_.iloc[rd.randint(0,7000-1)].values
    
    #grafico
    img = claseRep.reshape((28, 28)) #reordeno los pixeles
    ax.imshow(img, cmap='grey')
    ax.set_title(f"Clase {clase}",fontsize=26)
    ax.axis('off')        
        
plt.tight_layout()
plt.show()

# %% quiero ver cual es el valor maximo de intensidad que toman los pixeles
consulta = """
           SELECT MAX("1")
           FROM archivo2
           """
MaximoValorPixel = dd.query(consulta).df()
print(MaximoValorPixel) #Veo que los pixeles toman como mucho valor de intensidad 255 


# %%Armo un heat map que me muestre la mediana del valor que toma cada pixel. lo hago para cada clase.

#armo la figura
fig, axes = plt.subplots(2, 5, figsize=(26,9))
fig.suptitle("HeatMap de Mediana por Clase",y=1, fontsize=22)
axes = axes.flatten()

#armo una funcion para calcular todas las medianas de cada pixel por clases
def SacarMedianas(n):
    '''funcion para sacar la mediana de una clase n'''
    consulta = f""" 
                SELECT (*) 
                FROM archivo2 
                WHERE label = {n}
               """
    clase_ = dd.query(consulta).df()
    if len(archivo2)>785: #tengo el cuidado de quitar la label si es que sigue estando
        clase_=clase_.drop('label', axis=1)

    #saco la mediana de cada columna/pixel
    medianas_serie = clase_.median(numeric_only=True)
    medianas = medianas_serie.to_numpy()    

    return medianas

#bucle para ir generando los graficos
for i, label in enumerate(range(10)):
    ax = axes[i] #selecciono el espacio del grafico
    
    medianas=SacarMedianas(i) #calculo las medianas
    
    img = medianas.reshape((28, 28)) #grafico el heatmap
    im = ax.imshow(img, cmap='grey')
    cbar = ax.figure.colorbar(im, ax=ax)
    
    cbar.ax.set_ylabel("Valor", rotation=-90, va="bottom")  #agrego la barra de valores
    ax.set_title(f"Clase {label}",fontsize=17)
    ax.axis('off') 


plt.tight_layout()
plt.show()


# %%
#ahora me gustaria sacar la diferencia entre las medianas de dos clases

def diferenciaEntreClases(n,m):
    '''n:int (entre 0 y 9), m:int (entre 0 y 9)
    funcion para sacar la diferencia absoluta entre dos clases n y m.'''
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle(f"diferencia entre clase {n} y {m}",y=1, fontsize=22)
    axes = axes.flatten()

    medianas_n=SacarMedianas(n)
    medianas_m=SacarMedianas(m)
    Diferencias=abs(medianas_n-medianas_m)
    
    ValorDiferenciaEntreClases=Diferencias.sum()
    print(f'la suma de las diferencias absoluta de pixeles entre las clases {n} y {m} es: {ValorDiferenciaEntreClases}')

    grid_ticks = np.arange(-0.5, 28, 1) # De -0.5 a 27.5 para 28x28 píxeles

    img = medianas_n.reshape((28,28))
    im = axes[0].imshow(img, cmap='gray') 
    #axes[0].axis('off')
    axes[0].set_xticks(grid_ticks)
    axes[0].set_yticks(grid_ticks)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].grid(True, color='white', linewidth=0.5)
    axes[0].set_title(f"Clase {n}",fontsize=17)

    img = Diferencias.reshape((28,28))
    im = axes[1].imshow(img, cmap='hot') 
    #axes[1].axis('off')
    axes[1].set_xticks(grid_ticks)
    axes[1].set_yticks(grid_ticks)
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].grid(True, color='white', linewidth=0.5)
    #agrego la barra de valores
    cbar = axes[1].figure.colorbar(im, ax=axes[1])
    cbar.ax.set_ylabel("valor absoluto de la diferencia", rotation=-90, va="bottom")


    img = medianas_m.reshape((28,28))
    im = axes[2].imshow(img, cmap='gray') 
    #axes[2].axis('off')
    axes[2].set_xticks(grid_ticks)
    axes[2].set_yticks(grid_ticks)
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])
    axes[2].grid(True, color='white', linewidth=0.5)
    axes[2].set_title(f"Clase {m}",fontsize=17)

   
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()

#aplico la funcion en las clases de interes y grafico:    
diferenciaEntreClases(1,2)
diferenciaEntreClases(2,6)
# %% ahora me pregunto: y si quisiese graficar las 'concidencias' entre las medianas de dos clases?
#voy a armar una funcion que dependiendo de un parametro p
#tome como coincidencias todos los pixeles tales que
#la diferencia del valor entre ellos sea menor a p.
# de esta forma puedo elegir cual es el valor maximo de diferencia que considero como 'valores parecidos'
def coincidenciasEntreClases(n,m,p):
    '''
    n:int (entre 0 y 9), m:int (entre 0 y 9). p:float.
    funcion para sacar las coincidencias  entre dos clases n y m. Dependiendo de un parametro p'''
    fig, axes = plt.subplots(1, 3, figsize=(16,5)) #armo la figura
    fig.suptitle(f"coincidencias entre clase {n} y {m}",y=1, fontsize=22)
    axes = axes.flatten()

    medianas_n=SacarMedianas(n)
    medianas_m=SacarMedianas(m)
    
    #armo una funcion para armar el array de coincidencias
    def parecidos(a,b,p):
        '''a:array, b:array, f:float.
        Recorre los pixeles y busca los parecidos en base a un parametro p.
        Si la diferencia entre pixeles es menor a p, los tomo cuento como parecidos'''
        lista=[]
        for i in range(784):
                if abs(a[i]-b[i])<p:
                    lista.append(1)
                else:
                    lista.append(0)
        return np.array(lista)
    parecidos = parecidos(medianas_m,medianas_n,p) #aplico la funcion a las medianas de las dos clases m y n
    SumaParecidos=parecidos.sum()
    titulo_coincidencias = f"Píxeles Coincidentes: {SumaParecidos} (p < {p})"
    print(f'(tomando como parecidos cuya diferencia sea menor a {p})') 
    print(f'la suma de las coincidencias de pixeles entre las clases {n} y {m} es: {SumaParecidos}') #cuantitativamente veo por la terminal cuantos pixeles coinciden

    #armo los graficos:
    grid_ticks = np.arange(-0.5, 28, 1) # De -0.5 a 27.5 para 28x28 píxeles 
    
    #grafico de n
    img = medianas_n.reshape((28,28))
    im = axes[0].imshow(img, cmap='gray') 
    #axes[0].axis('off')
    axes[0].set_xticks(grid_ticks)
    axes[0].set_yticks(grid_ticks)
    axes[0].set_xticklabels([])
    axes[0].set_yticklabels([])
    axes[0].grid(True, color='white', linewidth=0.5)
    axes[0].set_title(f"Clase {n}",fontsize=17)
    
    #grafico de las coincidencias:
    colors = ['white','blue']
    cmap = mcolors.ListedColormap(colors)
    
    img = parecidos.reshape((28,28))
    im = axes[1].imshow(img, cmap=cmap) 
    #axes[1].axis('off')
    axes[1].set_xticks(grid_ticks)
    axes[1].set_yticks(grid_ticks)
    axes[1].set_xticklabels([])
    axes[1].set_yticklabels([])
    axes[1].grid(True, color='white', linewidth=0.5)
    axes[1].set_title(titulo_coincidencias, fontsize=17)
    #agrego la barra de valores
   
    cbar = axes[1].figure.colorbar(im, ax=axes[1], ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['No Coincide', 'Coincide'] , fontsize=12, rotation=-90, va='center')    

    img = medianas_m.reshape((28,28))
    im = axes[2].imshow(img, cmap='gray') 
    #axes[2].axis('off')
    axes[2].set_xticks(grid_ticks)
    axes[2].set_yticks(grid_ticks)
    axes[2].set_xticklabels([])
    axes[2].set_yticklabels([])
    axes[2].grid(True, color='white', linewidth=0.5)
    axes[2].set_title(f"Clase {m}",fontsize=17)

    
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()
#aplico la funcion para graficar las clases de interes
coincidenciasEntreClases(2,6,50)
coincidenciasEntreClases(2,1,50)
coincidenciasEntreClases(2,6,1)
coincidenciasEntreClases(2,1,1)

# %%Quiero comparar imagenes de una misma clase que tan parecidos son/que tanto varian.
#defino una funcion para encontrar la media y el std
def Media_y_Std_Por_Pixel(dataset):
    """
    Calcula la Media y la Desviación Estándar
    para CADA UNO de los 784 píxeles dentro de un dataset.
    """
    # Media por pixel
    media_pixeles = dataset.mean(numeric_only=True).to_numpy()
    
    # Desviación estándar por pixel. Indica La variabilidad por píxel
    std_pixeles = dataset.std(numeric_only=True).to_numpy()
    
    return media_pixeles, std_pixeles


def VariacionPorClase(clase):
    '''clase:int(entre 0 y 9)
    funcion para graficar la variacion de los pixeles de una clase con un heatmap'''
    #selecciono la clase
    consulta = f"""
               SELECT *
               FROM archivo2
               WHERE label = {clase}
               """
    clase_= dd.query(consulta).df()
    clase_.drop('label', axis=1, inplace=True)

    #aplico la funcion 'Mdia_y_Std_Por_Pixel' a la clase de interes
    media, std = Media_y_Std_Por_Pixel(clase_) 
    #quiero saber cuales son los pixeles con mayor variabilidad
    print(f'los pixeles con mayor variabilidad en la clase {clase} son:')
    for indice, valor in enumerate(std):
        if valor > 111:
           print(f'{indice}:{valor}')
           
    #armo la figura del grafico
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Análisis de Variabilidad de la Clase {clase}", y=1.02, fontsize=20)
    axes = axes.flatten()

    #convierto los array de pixeles a imagen con un reshape
    img_media = media.reshape((28, 28))
    img_std = std.reshape((28, 28))

    #Grafico la media
    im0 = axes[0].imshow(img_media, cmap='magma')
    axes[0].set_title(" Imagen Promedio (Media de Píxeles)", fontsize=16)
    axes[0].axis('off')

    # barra de color
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', shrink=0.8)
    cbar0.ax.set_ylabel("Intensidad de Píxel (0-255)", rotation=-90, va="bottom")


    #Grafico la Desviación Estándar
    im1 = axes[1].imshow(img_std, cmap='viridis') 
    axes[1].set_title(" Mapa de Desviación Estándar (STD)", fontsize=16)
    axes[1].axis('off')

    #barra de color
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.8)
    cbar1.ax.set_ylabel("Desviación Estándar (Variación)", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.show()
    
VariacionPorClase(8)
#veo que 
# %% Y si saco la desviacion estandar para todo el dataset? 
#osea todas las clases juntas? hay algun valor que no varie? 
#osea que no aporte informacion y pueda descartar?

#reutilizo mi funcion Media_y_Std_Por_Pixel y la aplico a todo el dataset entero
def variacionDelDataset():
    if len(archivo2)>785: #tengo el cuidado de quitar la label si es que sigue estando
        archivo2_sinlabel=archivo2.drop('label', axis=1)
        
    media_dataset, std_dataset = Media_y_Std_Por_Pixel(archivo2_sinlabel)
    
    #armo la figura del grafico
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Análisis de Variabilidad del Dataset ", y=1.02, fontsize=20)
    axes = axes.flatten()

    #convierto los array de pixeles a imagen con un reshape
    img_media = media_dataset.reshape((28, 28))
    img_std = std_dataset.reshape((28, 28))

    #Grafico la media
    im0 = axes[0].imshow(img_media, cmap='magma')
    axes[0].set_title(" Imagen Promedio (Media de Píxeles)", fontsize=16)
    axes[0].axis('off')

    # barra de color
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', shrink=0.8)
    cbar0.ax.set_ylabel("Intensidad de Píxel (0-255)", rotation=-90, va="bottom", fontsize=12)


    #Grafico la Desviación Estándar
    im1 = axes[1].imshow(img_std, cmap='viridis') 
    axes[1].set_title(" Mapa de Desviación Estándar (STD)", fontsize=16)
    axes[1].axis('off')

    #barra de color
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.8)
    cbar1.ax.set_ylabel("Desviación Estándar (Variación)", rotation=-90, va="bottom", fontsize=12)

    plt.tight_layout()
    plt.show()

variacionDelDataset()

#veo por el grafico que solo las esquinas tienen poca variabilidad. Podria descartarlas.
#Al ser pocos pixeles descartarlos quizas no tiene tanto impacto.

# %%

#Hago un heatmap de desviacion estandar para comparar diferencias entre dos clases:
#reutilizando casi que la misma funcion "variacionDelDataset".
def variaciones_entre_clases(clase1,clase2):
    ''' '''
 #selecciono dos clases de un dataset(junto toda la informacion de ambas en un solo dataset, lo trabajo como si fuese una unica clase del punto anterior)
    consulta = f"""
               SELECT *
               FROM archivo2
               WHERE label = {clase1} OR label = {clase2}
               """ 
    clases= dd.query(consulta).df()
    clases.drop('label', axis=1, inplace=True)

    #aplico la funcion 'Mdia_y_Std_Por_Pixel' a las clases de interes
    media, std = Media_y_Std_Por_Pixel(clases) 
    
    #quiero saber cuales son los pixeles con mayor variabilidad
    print(f'los pixeles con mayor variabilidad entre clase {clase1} y clase {clase2} son:')
    for indice, valor in enumerate(std):
        if valor > 111:
           print(f'{indice}:{valor}')
    Sumastd=std.sum()/784 #suma promedio de la std de todos los pixeles 
    
    print(f'la desviacion estandar entre las clases {clase1} y {clase2} es: {Sumastd}')
    #armo la figura del grafico
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Análisis de Variabilidad de las Clases {clase1} y {clase2}", y=1.02, fontsize=20)
    axes = axes.flatten()

    #convierto los array de pixeles a imagen con un reshape
    img_media = media.reshape((28, 28))
    img_std = std.reshape((28, 28))

    #Grafico la media
    im0 = axes[0].imshow(img_media, cmap='magma')
    axes[0].set_title(" Imagen Promedio (Media de Píxeles)", fontsize=16)
    axes[0].axis('off')

    # barra de color
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', shrink=0.8)
    cbar0.ax.set_ylabel("Intensidad de Píxel (0-255)", rotation=-90, va="bottom")


    #Grafico la Desviación Estándar
    im1 = axes[1].imshow(img_std, cmap='viridis') 
    axes[1].set_title(" Mapa de Desviación Estándar (STD)", fontsize=16)
    axes[1].axis('off')

    #barra de color
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.8)
    cbar1.ax.set_ylabel("Desviación Estándar (Variación)", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.show()
    
    
variaciones_entre_clases(2,1)
variaciones_entre_clases(2,6)

# %%

#Ahora quiero hacer un hetamap de desviacion estandar pero con las medias de cada clase directamente. 
#la idea es complementar el analisis que hicimos de las coincidencias y diferencias entre las medianas anteriormente pero esta vez viendo las variabilidad
def variaciones_entre_medianas(n,m):
    ''' '''
    #saco las medianas de cada clase
    clase1=SacarMedianas(n)
    clase2=SacarMedianas(m)
    #las convierto a series
    clase1 = pd.Series(clase1)
    clase2 = pd.Series(clase2)
    #concateno ambas series
    clases1y2= pd.concat([clase1, clase2], axis=1)
    
    # Calculo la media y std a lo largo de las columnas
    media = clases1y2.mean(axis=1, numeric_only=True)
    std = clases1y2.std(axis=1, numeric_only=True)
    
    #lo paso a array
    media=media.to_numpy()
    std=std.to_numpy()
    
    #armo la figura del grafico
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"Análisis de Variabilidad de las Clases {n} y {m} (medianas)", y=1.02, fontsize=20)
    axes = axes.flatten()

    #convierto los array de pixeles a imagen con un reshape
    img_media = media.reshape((28, 28))
    img_std = std.reshape((28, 28))

    #Grafico la media
    im0 = axes[0].imshow(img_media, cmap='magma')
    axes[0].set_title(" Imagen Promedio (Media de Píxeles)", fontsize=16)
    axes[0].axis('off')

    # barra de color
    cbar0 = fig.colorbar(im0, ax=axes[0], orientation='vertical', shrink=0.8)
    cbar0.ax.set_ylabel("Intensidad de Píxel (0-255)", rotation=-90, va="bottom")


    #Grafico la Desviación Estándar
    im1 = axes[1].imshow(img_std, cmap='viridis') 
    axes[1].set_title(" Mapa de Desviación Estándar (STD)", fontsize=16)
    axes[1].axis('off')

    #barra de color
    cbar1 = fig.colorbar(im1, ax=axes[1], orientation='vertical', shrink=0.8)
    cbar1.ax.set_ylabel("Desviación Estándar (Variación)", rotation=-90, va="bottom")

    plt.tight_layout()
    plt.show()


variaciones_entre_medianas(2,1)
variaciones_entre_medianas(2,6)

# %%

#graficos de apoyo para punto 2. Clasificacion binaria.

coincidenciasEntreClases(4,5,1)
diferenciaEntreClases(4, 5)

#notar zonas de mayor variabilidad
variaciones_entre_clases(4,5)

#notar los pixeles de mayor variabilidad
variaciones_entre_medianas(4,5)
VariacionPorClase(4)
VariacionPorClase(5)
# %%
#en esta funcion queremos ver cuales son los pixeles que mas varian entre ambas clases pero ademas que su viariacion en la misma clase sea significativa
def variabilidadFisher(clase1,clase2):
    consulta = f"""
                   SELECT *
                   FROM archivo2
                   WHERE label = {clase1}
                   """ 
    clase1= dd.query(consulta).df()
    clase1.drop('label', axis=1, inplace=True)

    consulta = f"""
                SELECT *
                FROM archivo2
                WHERE label = {clase2}
                """ 
    clase2= dd.query(consulta).df()
    clase2.drop('label', axis=1, inplace=True)

    media1, std1 = Media_y_Std_Por_Pixel(clase1) 
    media2, std2 = Media_y_Std_Por_Pixel(clase2) 
    v = (media1-media2)**2/(std1**2 + std2**2)

    #quiero saber cuales son los pixeles con mayor variabilidad
    print(f'los pixeles con mayor variabilidad entre clase {clase1} y clase {clase2} son:')
    for indice, valor in enumerate(v):
        if valor > 1:
           print(f'{indice}:{valor}')
           
variabilidadFisher(4,5)


#%%###################################################
#                   PUNTO 2
######################################################
'''En esta seccion de codigo buscamos entrenar un modelo knn para clasificacion binaria
de las clases 4 y 5. '''

#volvemos a abrir los archivos en una nueva variable para que no se pise con las modificaciones que hicimos en la seccion anterior.
kminst = pd.read_csv(carpeta+"kmnist_classmap_char.csv")
kuzu_full = pd.read_csv(carpeta+"kuzushiji_full.csv")

#%%
#seleccionamos las clases 4 y 5
consulta = """
           SELECT *
           FROM kuzu_full
           WHERE label = 4 OR label = 5
           """
kuzu_full_4_5 = dd.query(consulta).df()

X = kuzu_full_4_5.iloc[:, :-1].values

#y = etiquetas (última columna)
y = kuzu_full_4_5.iloc[:, -1].values

#separamos en train y test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train_imgs = X_train.reshape(-1, 28, 28)

#Mostramos las primeras 9 imágenes para verificar
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
#Vemos que los bordes suelen dar peor que los atributos más centrados
modelos_knnC = pd.DataFrame(columns=["Cantidad_atributos", "Atributos_usados", "Accuracy"])
#Bordes-bordes-medio-seguidos-cualquiercosa-atributos con mayor variabilidad-atributos con mayor variabilidad entre clases pero baja dentro de una misma (fisher score).
subconjuntos_de_atributos4 = [[0,27,28,55],[56,83,84,168],[17,62,92,120],[444,445,446,447],[578,612,499,702],[416,370,360,388],[94,122,149,150]]
subconjuntos_de_atributos5 = [[0,27,28,55,56],[56,83,84,168,783],[17,62,92,120,355],[444,445,446,447,448],[578,612,499,702,550],[416,151,370,360,388],[177,178,205,206,122]]
subconjuntos_de_atributos7 = [[0,27,28,55,56,83,84],[56,83,84,168,196,224,252],[17,62,92,120,264,148,404],[444,445,446,447,448,449,450],[578,612,499,702,123,389,12],[151,388,398,416,370,360,389],[149,178,206,150,94,205,122]]
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
modelos_knnC.to_csv('Acuracy_Atributos_modelosKnn.csv', index=False)

#%%
#Este código genera atributos aleatorios, entrena un knn en base a esos atributos (testea el mejor n
#en enes y usa ese), y guarda cómo le fue a cada modelo (accuracy) con sus atributos elegidos, junto con la cantidad
#de atributos que usó y el mejor n. 

n_atributos = [3,4,5,8,10,15,20]
modelos_knn = pd.DataFrame(columns=["Cantidad_atributos", "Atributos_usados", "Mejor_n", "Accuracy"])

for cant_atributos in n_atributos:
    subconjuntos_de_atributos = []
    for i in range(10):
        subconjunto = rd.sample(range(0, 784), cant_atributos)
        subconjuntos_de_atributos.append(subconjunto)
    for atributos in subconjuntos_de_atributos:
        mejor_accuracy = 0
        mejor_n = 2
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
modelos_knn.to_csv('Acuracy_CantAtributos_MejorK_modelosKnn.csv', index=False)
        
#%%###################################################
#                   PUNTO 3
######################################################
''' en esta seccion de codigo queremos hacer una clasificacion multiclase de todos las clases del dataset'''

# Separo los datos en train y test
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
# (!)Este codigo tardó 45 minutos en correr. Lo dejamos comentado por las dudas. 
#fijamos una profundidad maxima en 10 y luego iteramos para probar distintos valores de min_samples_split y min_samples_leaf
# lista = ['gini','entropy']
# data1 = pd.DataFrame(columns=['profundidad','criterio','min_samples_split','min_samples_leaf','presicion'])
# for i in range(1,11,3):
#     for j in range(2,20,5):
#         for k in range(1,11,3):
#             for elem in lista:
#                 arbol = DecisionTreeClassifier(criterion=elem,
#                 splitter='best',
#                 max_depth=i,
#                 min_samples_split=j,
#                 min_samples_leaf=k,
#                 max_features=None,
#                 random_state=2)

#                 scores = cross_val_score(arbol, X_train, y_train, cv=5)
#                 data1.loc[len(data1)]=[i,elem,j,k,scores.mean()]  
                

#(!)este codigo tardo aproximadamente 20 minutos en correr.
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

#graficamos
graf = plt.scatter(data2['min_samples_split'],data2['min_samples_leaf'], c=data2['Precisión'])
plt.xlabel('min_samples_split')
plt.ylabel('min_samples_leaf')
plt.colorbar()
plt.show()

cbar = plt.colorbar(graf)
cbar.set_label('Precisión')
#%% 3d)

#Utilizamos el test con nuestro modelo y sacamos el acuracy
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


print(classification_report(y_test, prediction))
print(confusion_matrix(y_test, prediction))
