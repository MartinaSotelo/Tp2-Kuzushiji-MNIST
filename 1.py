
"""
Grupo Import_milanesas
Integrantes: 
Dulio Joaquin, 
Risuleo Franco, 
Perez Sotelo Martina

"""

# importamos las librerias que vamos a utilizar
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#poner nombre de la carpeta
carpeta = "/home/martina/import_milanesas_TP02/"

#abrimos los archivos
archivo1= pd.read_csv(carpeta+"kmnist_classmap_char.csv")
archivo2= pd.read_csv(carpeta+"kuzushiji_full.csv")

# %%
# armo una visualizacion de varias imagenes:

# El índice inicial (del csv) para las imágenes
indice_inicial = 5

mapa_caracteres = dict(zip(archivo1['class'], archivo1['char']))

# creo la figura
fig, axes = plt.subplots(
    nrows=5,      # 5 filas
    ncols=5,       # 5 columnas
    figsize=(10, 10)      # Tamaño de la figura 
)

axes = axes.flatten()

#recorro y visualizo las 10 imagenes a partir de mi indice_inicial
for k in range(25):
    fila_actual = indice_inicial + k
    etiqueta = archivo2.iloc[fila_actual, -1] #quiero la etiqueta de cada imagen
    datos_pixeles = archivo2.iloc[fila_actual, :-1].values
    img = datos_pixeles.reshape((28, 28))
    axes[k].imshow(img, cmap='gray')
    axes[k].set_title(f"Label: {etiqueta}", fontsize=10) #pongo de titulo la etiqueta a cada imagen
    axes[k].axis('off') 

plt.tight_layout()
plt.show()

# %%

# Paso 1: Separar X (píxeles) e y (etiquetas)
# Esto separa todas las 70000 filas de píxeles y las 70000 etiquetas
X = archivo2.iloc[:, :-1].values.astype('float32') / 255.0 # Normalizamos los píxeles a 0-1
y = archivo2.iloc[:, -1].values

# Paso 2: Crear la figura 
fig, axes = plt.subplots(1, 10, figsize=(15, 2))
plt.suptitle('Imagen Promedio por Clase (Kuzushiji-MNIST)', fontsize=16)

# Paso 3: Calcular y dibujar el promedio para cada clase
for clase in range(10):
    # a. Seleccionar solo las imágenes que pertenecen a la 'clase' actual
    imagenes_de_clase = X[y == clase]
    
    # b. Calcular la media a lo largo del eje 0 (es decir, el promedio de todas las imágenes)
    imagen_promedio = np.mean(imagenes_de_clase, axis=0)
    
    # c. Dibujar
    axes[clase].imshow(imagen_promedio.reshape(28, 28), cmap='gray')
    axes[clase].set_title(f"Clase {clase}", fontsize=10)
    axes[clase].axis('off')

plt.tight_layout()
plt.show()

# %%
#############################################################
# MUESTRA REPRESENTATIVA: UNA IMAGEN POR CLASE
############################################################


# Creamos un diccionario para guardar el índice de la primera imagen encontrada para cada clase.
indices_por_clase = {}
# Un array con las 10 posibles clases
clases_posibles = np.unique(y) # Esto devuelve [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Llenamos el diccionario buscando el primer índice para cada clase.
for clase in clases_posibles:
    # np.where(y == clase) devuelve una tupla, el [0] accede al array de índices.
    # [0] toma el primer índice de ese array (la primera imagen de esa clase).
    indices_por_clase[clase] = np.where(y == clase)[0][0]
    
# Crear la figura (1 fila, 10 columnas)
fig, axes = plt.subplots(1, 10, figsize=(18, 2))
plt.suptitle('Muestra Representativa: Una Imagen por Clase', fontsize=16)

for i, clase in enumerate(clases_posibles):
    # Obtener el índice guardado
    indice = indices_por_clase[clase]
    
    # Extraer y reformar la imagen (ya está en X)
    img = X[indice].reshape((28, 28))
    
    # Dibujar
    axes[i].imshow(img, cmap='gray')
    axes[i].set_title(f"Clase: {clase}", fontsize=12)
    axes[i].axis('off')

plt.tight_layout()
plt.show()


# %%

#######
# MUESTRA ALEATORIA DE 10 IMAGENES
#######
# --- PASO 1: Configuración ---
num_filas = 10
num_cols = 10
total_imagenes = num_filas * num_cols # 100 imágenes

# Generar 100 índices aleatorios únicos dentro del rango del dataset
# El tamaño del dataset es len(y) (ej: 70000)
indices_aleatorios = np.random.choice(len(y), size=total_imagenes, replace=False)

# --- PASO 2: Crear la Figura ---
fig, axes = plt.subplots(
    nrows=num_filas,
    ncols=num_cols,
    figsize=(12, 12) # Cuadrícula más grande
)
axes = axes.flatten() # Aplanar para facilitar la iteración
plt.suptitle('Muestra Aleatoria de 100 Imágenes', fontsize=16, y=1.02) # y ajusta la altura del título

# --- PASO 3: Iterar y Dibujar ---
for k in range(total_imagenes):
    
    # 1. Obtener el índice aleatorio para esta posición
    indice_actual = indices_aleatorios[k]
    
    # 2. Extraer la etiqueta y los píxeles
    etiqueta = y[indice_actual]
    img = X[indice_actual].reshape((28, 28))
    
    # 3. Dibujar
    axes[k].imshow(img, cmap='gray')
    axes[k].set_title(f"Label: {etiqueta}", fontsize=8) # Título más pequeño
    axes[k].axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.98]) # Ajustar el layout para no cortar el suptitle
plt.show()
# %%
# COMPARACION DE 10 EJEMPLOS ALEATORIOS POR CLASE

# Parámetros
clases_posibles = np.unique(y) # [0, 1, 2, ..., 9]
num_clases = len(clases_posibles) # 10
num_ejemplos_por_clase = 10

# --- PASO 1: Crear la Figura ---

fig, axes = plt.subplots(
    nrows=num_clases,
    ncols=num_ejemplos_por_clase,
    figsize=(15, 15)
)
plt.suptitle('Comparación de 10 Ejemplos ALEATORIOS por Clase', fontsize=20, y=1.02)


# --- PASO 2: Iterar por Clase y Muestrear Aleatoriamente ---

for i, clase in enumerate(clases_posibles):
    
    # 1. Encontrar todos los índices de las imágenes que pertenecen a esta clase
    indices_clase_total = np.where(y == clase)[0]
    
    # 2. SELECCIONAR 10 ÍNDICES AL AZAR de ese grupo
    # np.random.choice toma un array (los índices) y selecciona 'size' elementos de él.
    # 'replace=False' asegura que no elijamos la misma imagen dos veces.
    
    # Manejar el caso si la clase tiene menos de 10 ejemplos (aunque K-MNIST es balanceado)
    num_a_elegir = min(num_ejemplos_por_clase, len(indices_clase_total))
    
    indices_a_mostrar = np.random.choice(
        indices_clase_total, 
        size=num_a_elegir, 
        replace=False
    )
    
    # 3. Iterar sobre los 10 índices seleccionados al azar para dibujar la fila
    for j, indice in enumerate(indices_a_mostrar):
        
        # Extraer los datos de píxeles y reformar
        img = X[indice].reshape((28, 28))
        
        # Dibujar en la posición (i, j)
        axes[i, j].imshow(img, cmap='gray')
        
        # Poner la etiqueta solo en la primera columna
        if j == 0:
            axes[i, j].set_title(f"Label {clase}", fontsize=12, fontweight='bold', loc='left')
        
        # Eliminar ejes y ticks
        axes[i, j].axis('off')
        
    # Rellenar los espacios vacíos si la clase tenía menos de 10 ejemplos
    for j in range(num_a_elegir, num_ejemplos_por_clase):
        axes[i, j].axis('off')


# Asegurar que los subgráficos no se superpongan
plt.tight_layout(rect=[0, 0, 1, 0.98])
plt.show()