# PlanningWithSubgoals

Repositorio que contiene el código y la documentación asociada al proyecto de
aprendizaje de objetivos utilizando planificación.

## Carlos

Por ahora, he obtenido un dataset y he entrenado el modelo. Posteriormente integraré el modelo en la arquitectura del sistema.

El modelo predice la longitud del plan necesario para conseguir un subobjetivo, dado un estado inicial. La entrada está codificada como una matriz de observaciones, donde cada posición representa una casilla del nivel. Para representar el objeto que se encuentra en cada casilla he usado una codificación one-hot: un vector de 0 donde la posición del único 1 depende del tipo de objeto que se encuentre en esa casilla. Para representar la posición del subobjetivo he usado la misma codificación: se representa como un 1 en la última posición del vector one-hot de la posición de la matriz asociada a la casilla donde se encuentra dicho subobjetivo. La salida del modelo es un número real, que se corresponde con la predicción de la longitud del plan asociado a ese estado inicial y subobjetivo.

Los datasets usados están dentro de la carpeta [Datasets](../datasets/). Para obtener el dataset de entrenamiento, que consta de 5000 ejemplos, ejecuté el agente en los niveles 0, 1 y 2 (sin enemigos). Al comienzo de cada nivel, así como cuando se consigue cada gema, se crea un plan para cada uno de los posibles subobjetivos (gemas) elegibles en ese estado. De entre todos esos planes, se elige uno al azar y se ejecuta, terminando el agente en la posición del subobjetivo correspondiente, momento en el cual se repite el proceso. Esto se repite una y otra vez, resolviendo cada nivel de forma sucesiva hasta conseguir 5000 muestras para el dataset. Cada muestra viene dada por una pareja (estado_y_subobjetivo, longitud plan). El dataset de validación consta de 400 ejemplos y lo obtuve de la misma forma, pero esta vez ejecutando el agente sobre los niveles 3 y 4.

Para implementar los modelos he usado **_TensorFlow_**. Primero hice unas pruebas con una red neuronal feed-forward y después pasé a usar una CNN, que me dio mejores resultados. El código de ambos modelos se encuentra en el repositorio, en la forma de un **_Jupyter Notebook_**:

- [Feed-Forward NN](/tree/master/Models/ModeloPlanificacion.ipynb)
- [CNN](../Models/ModeloCNN.ipynb)

Para visualizar los resultados he usado una herramienta que viene por defecto con _TensorFlow_: _**TensorBoard**_. Esta herramienta permite guardar la arquitectura y las métricas del modelo y visualizarlas en forma de gráficos. Los logs de las diversas pruebas con los modelos están guardados en las carpetas [ModeloCNN_log](../Models/ModeloCNN_log/) y [ModeloPlanificacion_log](../Models/ModeloCNN_log/).

#### Tutorial

Para ejecutar el código de los modelos lo primero es instalar los paquetes necesarios: _TensorFlow_ (_TensorBoard_ creo que viene incluido por defecto) y _Jupyter_.
1. Primero activamos el **entorno de _Conda_** que queremos usar: `conda activate <nombre_entorno>` en Linux o `activate <nombre_entorno>` en Windows.
2. Instalamos los paquetes: `conda install tensorflow` y `conda install jupyter`.

		Nota: Jupyter ya debería venir instalado por defecto con Anaconda, por lo que no
        debería ser necesario volverlo a instalar.
             
3. Abrimos un terminal de forma que el directorio de trabajo sea la carpeta donde se encuentran los cuadernos de _Jupyter_ (archivos _.ipynb_) y ejecutamos el siguiente comando: `jupyter notebook`
4. Se nos abrirá una pestaña en el navegador. Seleccionamos el cuaderno que queremos ejecutar y ejecutamos todas las celdas: _Menú / Cells / Execute All_. Para más información sobre el uso de _Jupyter_, recomiendo leer la [documentación oficial](https://jupyter-notebook.readthedocs.io/en/stable/).
5. Para visualizar los _logs_ usamos _TensorBoard_. Para ejecutarlo, abrimos un terminal y escribimos: `tensorboard --logdir=path/to/log-directory --host=localhost`.
6. Abrimos el navegador y abrimos la dirección _localhost:6006_.
7. La pestaña _Graph_ muestra una visualización del modelo. La pestaña _Scalars_ muestra las gráficas de la función de pérdida en el entrenamiento y validación. Para más información sobre el uso de _TensorBoard_, recomiendo echarle un vistazo a este [vídeo](https://www.youtube.com/watch?v=eBbEDRsCmv4&list=PLPl9hCpYCVfPI3GG99vALTZlK0AaaFmls) de Youtube del canal oficial de Google.
	

## Vladis

Planificador utilizado: [LAMA](https://github.com/rock-planning/planning-lama)
