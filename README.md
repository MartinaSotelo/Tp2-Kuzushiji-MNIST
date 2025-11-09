# Tp2-Kuzushiji-MNIST
Este proyecto realiza un análisis exploratorio detallado del dataset Kuzushiji-MNIST y evalúa el rendimiento de dos modelos de Machine Learning: k-Nearest Neighbors (k-NN) para clasificación binaria y Árboles de Decisión para clasificación multiclase.
El enfoque se centra en la selección de atributos (píxeles) basada en su variabilidad y poder discriminatorio

Librerias necesarias para la ejecucion:
* pandas -> Manipulación de datos
* numpy -> Operaciones 
* duckdb -> Consultas a DataFrames
* scikit-learn -> Modelos de Machine Learning: kNN, Árbol de Decisión y Métricas
* matplotlib -> Generación de gráficos

Se debe completar la variable carpeta (linea 38) con el path correspondiente de donde se encuentre el archivo .py

El codigo se encuentra separado en tres secciones principales: 
* Punto 1 -> exploramos, vemos caracteristicas, estructura y realizamos un analisis exploratorio del dataset apoyandonos de graficos. Contiene:
      - Representación Visual -> Muestra una imagen representativa por cada clase (0-9).
      - Heatmap de Medianas	-> Muestra el patrón de intensidad medio de cada clase.
      - Diferencia de Medianas -> Mide la diferencia absoluta entre las medianas de dos clases.
      - Análisis de Coincidencias	-> 	Identifica píxeles "similares" basados en un parametro de diferencia p.
      - Variabilidad por clase ->	Calcula y grafica la Desviación Estándar (STD) dentro de una única clase.
      - Variabilidad del dataset ->	Calcula y grafica la Desviación Estándar (STD) dentro del dataset entero.
      - Variabilidad entre clases ->	Calcula y grafica la Desviación Estándar (STD) dentro de dos clases.
      - Fisher Score -> Calcula el fisher score para identificar los píxeles con mayor poder discriminante entre dos clases.
  
* Punto 2 -> clasificación binaria de las Clases 4 y 5.
* 
* Punto 3 -> clasificación multiclase de todas las clases.

Autores: Dulio Joaquin, Risuleo Franco, Perez Sotelo Martina
