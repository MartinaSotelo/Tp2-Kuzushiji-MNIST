# Tp2-Kuzushiji-MNIST
Este proyecto realiza un análisis exploratorio detallado del dataset Kuzushiji-MNIST y evalúa el rendimiento de dos modelos de Machine Learning: k-Nearest Neighbors (k-NN) para clasificación binaria y Árboles de Decisión para clasificación multiclase.
El enfoque se centra en la selección de atributos (píxeles) basada en su variabilidad y poder discriminatorio

Librerias necesarias para la ejecución:
* pandas -> Manipulación de datos
* numpy -> Operaciones 
* duckdb -> Consultas a DataFrames
* scikit-learn -> Modelos de Machine Learning: kNN, Árbol de Decisión y Métricas
* matplotlib -> Generación de gráficos

Se debe completar la variable carpeta (linea 38) con el path correspondiente de donde se encuentre el archivo .py

El código se encuentra separado en tres secciones principales: 

* Punto 1 -> exploramos, vemos características, estructura y realizamos un análisis exploratorio del dataset apoyandonos de          gráficos. Contiene:
      - Representación Visual       ->       Muestra una imagen representativa por cada clase (0-9).
      - Heatmap de Medianas	      ->       Muestra el patrón de intensidad medio de cada clase.
      - Diferencia de Medianas      ->       Mide la diferencia absoluta entre las medianas de dos clases.
      - Análisis de Coincidencias	->       Identifica píxeles "similares" basados en un parametro de diferencia p.
      - Variabilidad por clase      ->       Calcula y grafica la Desviación Estándar (STD) dentro de una única clase.
      - Variabilidad del dataset    ->	   Calcula y grafica la Desviación Estándar (STD) dentro del dataset entero.
      - Variabilidad entre clases   ->	   Calcula y grafica la Desviación Estándar (STD) dentro de dos clases.
      - Fisher Score                ->       Calcula el fisher score para identificar los píxeles con mayor poder discriminante                                                 entre dos clases.
  
* Punto 2 -> clasificación binaria de las Clases 4 y 5 (K-NN).
  Se utiliza el clasificador k-Nearest Neighbors para distinguir entre las clases 4 y 5. El foco está en la selección de distintos   subconjuntos de píxeles. Contiene:
      - Datos -> Se filtra el dataset para incluir solo las etiquetas 4 y 5, y se realiza el train_test_split.
      - Experimento con Atributos Fijos (modelos_knnC) -> se evalúa el rendimiento de knn (con k=5) utilizando subconjuntos de             píxeles elegidos estratégicamente (bordes,centro,píxeles de alto Fisher Score, etc).
        (Resultados guardados en Acuracy_Atributos_modelosKnn.csv)
      - Experimento con Atributos Aleatorios y Búsqueda de k (modelos_knn) -> Prueba distintos tamaños de subconjuntos de píxeles          elegidos al azar. Para cada subconjunto, busca el mejor valor de k (entre 2 y 30 con paso 5).
        (Resultados guardados en Acuracy_CantAtributos_MejorK_modelosKnn.csv)
  
* Punto 3 -> clasificación multiclase de todas las clases. Contiene:
      - Ajuste de Profundidad (3b): Se utiliza cross_val_score para determinar la profundidad óptima (max_depth) del                       árbol,variando de 1 a 10.
      - Búsqueda de Hiperparámetros (3c): Se realiza una búsqueda (puede ser lenta, por eso está comentada una parte) para                 optimizar: Criterio de división (gini o entropy), min_samples_split y min_samples_leaf.
      - Análisis Final (3d): Se entrena el modelo final con los mejores hiperparámetros (ej. max_depth=10, criterion='entropy'), se evalúa su rendimiento en el conjunto de prueba (X_test), mostrando: 
        classification_report: Métricas por clase (Precisión, Recall, F1-Score)
        confusion_matrix: Visualización de las confusiones entre las 10 clases.

Las tablas resultadas del punto 2 y 3 se guardan en carpetas resultado_binario y resultado_multiclase respectivamente.

Autores: Dulio Joaquin, Risuleo Franco, Perez Sotelo Martina
