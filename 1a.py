
"""
Grupo Import_milanesas
Integrantes: 
Dulio Joaquin, 
Risuleo Franco, 
Perez Sotelo Martina

exploracion del dataset

"""

# importamos las librerias que vamos a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import duckdb as dd
import matplotlib.colors as mcolors
#poner nombre de la carpeta
carpeta = "/home/martina/import_milanesas_TP02/"

#leemos los archivos
archivo1= pd.read_csv(carpeta+"kmnist_classmap_char.csv")
archivo2= pd.read_csv(carpeta+"kuzushiji_full.csv")

# %%
#quiero ver cual es el valor maximo de intensidad que toman los pixeles
consulta = """
           SELECT MAX("1")
           FROM archivo2
           """
MaximoValorPixel = dd.query(consulta).df()
print(MaximoValorPixel)
#Veo que los pixeles toman como mucho valor de intensidad 255 

# %%
# Armo un heat map que me muestre la mediana del valor(escala blanco-negro) que toma cada pixel. Para cada clase.

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
    clase_.drop('label', axis=1, inplace=True) #elimino la label

    #saco la mediana de cada columna/pixel
    medianas_serie = clase_.median(numeric_only=True)
    medianas = medianas_serie.to_numpy()    

    return medianas

#bucle para ir generando los graficos
for i, label in enumerate(range(10)):
    ax = axes[i] #selecciono el espacio del grafico
    
    medianas=SacarMedianas(i) #calculo las medianas
    
    img = medianas.reshape((28, 28)) #grafico el heatmap
    im = ax.imshow(img, cmap='viridis')
    cbar = ax.figure.colorbar(im, ax=ax)
    
    cbar.ax.set_ylabel("Valor", rotation=-90, va="bottom")  #agrego la barra de valores
    ax.set_title(f"Clase {label}",fontsize=17)
    ax.axis('off') 


plt.tight_layout()
plt.show()

# %%
#ahora me gustaria sacar la diferencia entre las medianas de dos clases

#armo una funcion
def diferenciaEntreClases(n,m):
    '''funcion para sacar la diferencia absoluta entre dos clases n y m.'''
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
    fig.suptitle(f"diferencia entre clase {n} y {m}",y=1, fontsize=22)
    axes = axes.flatten()

    medianas_n=SacarMedianas(n)
    medianas_m=SacarMedianas(m)
    Diferencias=abs(medianas_n-medianas_m)

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
    
diferenciaEntreClases(1, 2)
diferenciaEntreClases(2,6)
# %%
#y si quisiese graficar las concidencias entre las medianas de dos clases?

#armo una funcion
def coincidenciasEntreClases(n,m,p):
    '''funcion para sacar las coincidencias  entre dos clases n y m.'''
    fig, axes = plt.subplots(1, 3, figsize=(16,5))
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
                    lista.append(a[i])
                else:
                    lista.append(0)
        return np.array(lista)
    parecidos = parecidos(medianas_m,medianas_n,p)


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

    img = parecidos.reshape((28,28))
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
    
coincidenciasEntreClases(2,6,40)
coincidenciasEntreClases(2,1,40)
