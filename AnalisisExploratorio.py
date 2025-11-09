
"""
Grupo Import_milanesas
Integrantes: 
Dulio Joaquin, 
Risuleo Franco, 
Perez Sotelo Martina

Exploracion del dataset
Este archivo contiene el codigo de todos los graficos presentes en el 
de la sección 'analisis exploratorio de los datos''

"""
#%%
# importamos las librerias que vamos a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb as dd
import matplotlib.colors as mcolors
import random as rd

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
    fig.suptitle(f"Análisis de Variabilidad de las Clases {n} y {m}", y=1.02, fontsize=20)
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
