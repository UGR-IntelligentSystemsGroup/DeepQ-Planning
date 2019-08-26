# PlanningWithSubgoals

Repositorio que contiene el código y la documentación asociada al proyecto de
aprendizaje de objetivos utilizando planificación.

## Carlos

Por ahora, he obtenido un dataset y he entrenado el modelo. Posteriormente integraré el modelo en la arquitectura del sistema.

El modelo predice la longitud del plan necesario para conseguir un subobjetivo, dado un estado inicial. La entrada está codificada como una matriz de observaciones, donde cada posición representa una casilla del nivel. Para representar el objeto que se encuentra en cada casilla he usado una codificación one-hot: un vector de 0 donde la posición del único 1 depende del tipo de objeto que se encuentre en esa casilla. Para representar la posición del subobjetivo he usado la misma codificación: se representa como un 1 en la última posición del vector one-hot de la posición de la matriz asociada a la casilla donde se encuentra dicho subobjetivo. La salida del modelo es un número real, que se corresponde con la predicción de la longitud del plan asociado a ese estado inicial y subobjetivo.

Los datasets usados están dentro de la carpeta [Datasets](../../tree/master/datasets/). Para obtener el dataset de entrenamiento, que consta de 5000 ejemplos, ejecuté el agente en los niveles 0, 1 y 2 (sin enemigos). Al comienzo de cada nivel, así como cuando se consigue cada gema, se crea un plan para cada uno de los posibles subobjetivos (gemas) elegibles en ese estado. De entre todos esos planes, se elige uno al azar y se ejecuta, terminando el agente en la posición del subobjetivo correspondiente, momento en el cual se repite el proceso. Esto se repite una y otra vez, resolviendo cada nivel de forma sucesiva hasta conseguir 5000 muestras para el dataset. Cada muestra viene dada por una pareja (estado_y_subobjetivo, longitud plan). El dataset de validación consta de 400 ejemplos y lo obtuve de la misma forma, pero esta vez ejecutando el agente sobre los niveles 3 y 4.

Para implementar los modelos he usado **_TensorFlow_**. Primero hice unas pruebas con una red neuronal feed-forward y después pasé a usar una CNN, que me dio mejores resultados. El código de ambos modelos se encuentra en el repositorio, en la forma de un **_Jupyter Notebook_**:

- [Feed-Forward NN](../../tree/master/Models/ModeloPlanificacion.ipynb)
- [CNN](../../tree/master/Models/ModeloCNN.ipynb)

Para visualizar los resultados he usado una herramienta que viene por defecto con _TensorFlow_: _**TensorBoard**_. Esta herramienta permite guardar la arquitectura y las métricas del modelo y visualizarlas en forma de gráficos. Los logs de las diversas pruebas con los modelos están guardados en las carpetas [ModeloCNN_log](../../tree/master/Models/ModeloCNN_log/) y [ModeloPlanificacion_log](../../tree/master/Models/ModeloCNN_log/).

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
	
### Modelo Greedy (Nuevo)

Ahora describiré la arquitectura del agente con el modelo greedy (que solo planifica para el siguiente subobjetivo).
#### Arquitectura del modelo
La arquitectura de la red neuronal es, a grandes rasgos la siguiente:
1. Un input viene dado por una matriz one-hot de observaciones. Los inputs son normalizados usando **_batch normalization_**, tanto en la fase de entrenamiento como en la de validación. Esta técnica es la que ha permitido aumentar drásticamente el rendimiento del modelo. Esto es debido a que esta técnica permite conseguir que los inputs estén centrados en cero y con una desviación típica de uno (estén normalizados), a pesar de que no tenemos todo el dataset al comienzo del entrenamiento (es por esta razón que no se pueden normalizar los inputs de la forma clásica) sino que el dataset se va creando progresivamente con el conocimiento adquirido por el agente.
2. Después, se aplica una capa **_convolucional_**, con solo dos filtros.
3. A continuación, se aplica **_max pooling_**.
4. Tras eso, se usa una capa **_fully connected_**, con solo 16 unidades.
5. A la salida de esa capa, se le aplica **_dropout_** para combatir el _overfitting_.
6. Por último, la capa de salida consta de una sola neurona, que devuelve un número real. Esta es la predicción del número de acciones del plan.

La función de **pérdida** usada es simplemente la suma de **errores al cuadrado**. El optimizador usado para minimizar esta función (con descenso de gradientes) es **_Adam_**.

Todo el código relativo al modelo se encuentra implementado en el archivo [**_LearningModel.py_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/LearningModel.py). Los **_logs_** de las distintas pruebas realizadas se encuentran en la carpeta [ModelLogs](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/ModelLogs).

#### Tutorial de Ejecución
Al estar el modelo ya integrado en la arquitectura del agente, solo es necesario (a diferencia de en la fase anterior) ejecutar el agente a través del archivo **_oneclickRunFromPythonClient_**. 
El comportamiento del agente es el siguiente:

- En la fase de entrenamiento, elige los subobjetivos al azar, planifica para conseguirlos y guarda estos planes como experiencia para entrenar el modelo.
- Una vez en la fase de validación, usa el modelo entrenado para elegir los subobjetivos. Concretamente, cuando es necesario elegir el siguiente subobjetivo a conseguir, el agente escoge la gema para la cual el modelo predice el menor número de acciones para llegar hasta ella.

A pesar de que este sería el comportamiento "natural", la fase de entrenamiento lleva cierto tiempo (unos 20 minutos de media) para conseguir buenos resultados en los niveles de validación. Por este motivo, ya he ejecutado la fase de entrenamiento, entrenado el modelo y guardado dicho modelo dentro de la carpeta [**_SavedModels_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/SavedModels).
 De esta forma, el comportamiento del agente que se encuentra actualmente implementado es el siguiente:

1. Al iniciar la ejecución carga el modelo pre-entrenado.
2. Se salta la fase de entrenamiento (he cambiado el fichero [**_CompetitionParameters_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/utils/CompetitionParameters.py) para que sea así).
3. Ejecuta la fase de validación (niveles 3 y 4). Usa el modelo cargado (ya entrenado) para elegir los subobjetivos. Para que sea más sencillo medir su rendimiento, he hecho que imprima por el terminal el número de acciones que necesita para completar cada nivel.

### Modelo Deep Q Learning
Este modelo supone una mejora respecto a la arquitectura greedy ya que, a diferencia de esta, elige el siguiente subobjetivo teniendo en cuenta no solo el plan desde el estado actual hasta ese subobjetivo, sino también el resto del plan desde el subobjetivo hasta completar el nivel. Esto quiere decir que la arquitectura ya no es greedy. Para conseguir esto, he usado _**Deep Q-Learning**_.
#### Arquitectura del modelo
La arquitectura de la red neuronal es muy parecida a la del modelo greedy:
1. Inputs normalizados con **_batch normalization_**.
2. Capa **_convolucional_**, con **_max pooling_**.
3. Capa **_fully connected_**, con 64 unidades. _**Dropout**_ tras esta capa.
4. Otra capa **_fully connected_**, con 16 unidades. _**Dropout**_ también tras esta capa.
5. Capa de salida, formada por una única neurona. Devuelve la predicción sobre el número de acciones del plan, teniendo en cuenta el _**factor de descuento**_ (_gamma_): _output = num acciones subplan 1 + gamma\*num acciones subplan 2 + gamma^2\*num acciones subplan 3 ..._, donde _num acciones subplan i_ hace referencia al número de acciones del plan desde un subobjetivo al siguiente.

La función de **pérdida** usada es simplemente la suma de **errores al cuadrado**. El optimizador usado para minimizar esta función (con descenso de gradientes) es **_Adam_**.

Todo el código relativo al modelo se encuentra implementado en el archivo [**_LearningModel.py_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/LearningModel.py). Los **_logs_** de las distintas pruebas realizadas se encuentran en la carpeta [DQNetworkLogs](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/DQNetworkLogs).

#### Tutorial de Ejecución
Para ejecutar este modelo, solo hay que seguir los mismos pasos que para el modelo greedy. Solo es necesario hacer algunas aclaraciones:

- Los distintos modelos usados en las pruebas se encuentran guardados en la carpeta  [**_SavedModels_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/SavedModels), excepto por el mejor modelo de todos (el que se corresponde con la arquitectura final). Este modelo se encuentra dentro de esta carpeta, en la subcarpeta [**_FinalArchitecture_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/SavedModels/FinalArchitecture).
- El código tal y como se ha subido a github en la última versión, se salta la fase de entrenamiento, carga el modelo pre-entrenado (el que se corresponde con _FinalArchitecture_) y ejecuta la fase de validación. 
Si se quiere ejecutar la fase de entrenamiento, solo hay que seguir estos pasos:
	1. Comentar en _**Agent.py**_ la línea al final del método _**\_\_init\_\_**_ que carga el modelo pre-entrenado (_**self.model.load_model( ... )**_).
	2. Cambiar el tiempo de entrenamiento en el fichero [**_CompetitionParameters_**](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/utils/CompetitionParameters.py). Para ello, se puede simplemente descomentar la línea _TOTAL_LEARNING_TIME = 180*MILLIS_IN_MIN_, que establece el tiempo de entrenamiento en 3 horas (_y comentar la línea que se encuentra actualmente descomentada_).
	3. Es recomendable ejecutar  la simulación sin la visualización. Para ello, solo hace falta editar el fichero **_oneclickRunFromPythonClient_** y quitar el argumento **_-visuals_**.

	De esta forma, el código ejecutará la fase de entrenamiento. Por defecto, tras llegar a 575 y 1500 iteraciones del entrenamiento, el modelo se guardará con el nombre dado por _self.save_path_ y _self.save_path_2_ (atributos que se encuentran en el fichero _Agent.py_ en el método _\_\_init\_\__. El tiempo medio que tarda en llegar a las 575 iteraciones es de 45 minutos y se corresponde con unos 600 elementos del dataset.
	Mientras se está ejecutando, el modelo va guardando _logs_ acerca de la pérdida en el dataset de entrenamiento (experiencia que va recolectando mientras va entrenando). El nombre del _log_ viene dado por el parámetro _writer_name_, que se le pasa al constructor del DeepQModel, también en el método \_\_init\_\_ en Agent.py. Estos _logs_ se guardan en la carpeta [DQNetworkLogs](../../tree/master/GVGAI/clients/GVGAI-PythonClient/src/DQNetworkLogs) y se pueden visualizar usando _**TensorBoard**_, tal y como ya expliqué en el apartado del modelo greedy.

## Vladis

Planificador utilizado: [LAMA](https://github.com/rock-planning/planning-lama)
