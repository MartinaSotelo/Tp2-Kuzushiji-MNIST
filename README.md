README.md
Clasificación de Caracteres Kuzushiji-MNIST 🏯

Este proyecto realiza un análisis exploratorio profundo y aplica modelos de Machine Learning para la clasificación de caracteres japoneses antiguos (Kuzushiji-MNIST). El enfoque principal radica en la selección estratégica de atributos (píxeles) basada en variabilidad y poder discriminatorio (Fisher Score).

    ⚠️ Estado del Proyecto: En proceso de refinamiento. Actualmente se están analizando en profundidad las matrices de confusión y métricas por clase para optimizar la precisión de los modelos multiclase.

👥 Grupo: Import_Milanesas

    Dulio Joaquin

    Risuleo Franco

    Perez Sotelo Martina

🛠️ Tecnologías y Librerías

El proyecto está desarrollado íntegramente en Python 3 utilizando un único script modularizado:

    pandas & numpy: Manipulación y operaciones de datos.

    duckdb: Consultas SQL eficientes sobre DataFrames.

    scikit-learn: Implementación de modelos (k-NN, Decision Tree) y métricas.

    matplotlib: Visualización de heatmaps y análisis de variabilidad.

📋 Estructura del Proyecto

El código se divide en tres secciones principales ejecutables por fragmentos (#%%):
1. Análisis Exploratorio de Datos (EDA)

Investigación visual del dataset para entender cómo "ven" los modelos cada carácter:

    Visualización: Representación de imágenes aleatorias por clase.

    Heatmaps de Medianas: Identificación de patrones promedio de trazos.

    Estudio de Variabilidad: Uso de Desviación Estándar (STD) y Fisher Score para encontrar los píxeles más informativos entre clases críticas.

2. Clasificación Binaria (k-NN)

Foco en distinguir las clases 4 y 5:

    Experimentos con Atributos Fijos: Evaluación de píxeles en bordes vs. centro y zonas de alto Fisher Score.

    Búsqueda de Hiperparámetros: Optimización automática de la cantidad de atributos aleatorios y el valor de k (vecinos).

3. Clasificación Multiclase (Árboles de Decisión)

Clasificación de las 10 categorías del dataset:

    Ajuste de Profundidad: Uso de validación cruzada (cross_val_score) para evitar el sobreajuste (overfitting).

    Optimización: Búsqueda del mejor criterio (Gini vs. Entropía) y parámetros de poda (min_samples_split, min_samples_leaf).

🚀 Instrucciones de Ejecución

    Clonar el repositorio.

    Asegurarse de tener los archivos kmnist_classmap_char.csv y kuzushiji_full.csv en la misma carpeta.

    Configuración del Path: Modificar la línea 38 del script con la ruta local de tu directorio:
    Python

    carpeta = "TU_RUTA_AQUI/"

    Ejecutar las secciones según el análisis deseado. Los resultados se guardarán automáticamente en las carpetas /Resultados_Binario y /Resultados_Multiclase.

📈 Próximos Pasos

    [ ] Análisis exhaustivo de la Matriz de Confusión para identificar qué caracteres japoneses se confunden más frecuentemente.

    [ ] Refinar el cálculo de métricas (Precision, Recall y F1-Score) por clase para balancear el modelo multiclase.

Notas de diseño

    El código utiliza DuckDB para agilizar el filtrado de datos, permitiendo consultas SQL directamente sobre los archivos CSV cargados.

    Se priorizó la modularización para permitir la experimentación rápida con diferentes subconjuntos de píxeles.
