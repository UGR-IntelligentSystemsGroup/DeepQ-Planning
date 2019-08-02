> Qué es

El archivo "environment.yml" describe un entorno de Anaconda (conda environment). 
Este entorno es el que estoy usando para ejecutar el código. Usa python 3.5 e incluye los paquetes necesarios
(hasta la fecha) como tensorflow.

> Uso

Para importar este archivo y crear el entorno de conda, hay que seguir los siguientes pasos:

	1. Instalar Anaconda (preferiblemente la última versión): hay que descargar el script bash 
	    (en el caso de Linux) de la página "https://www.anaconda.com/distribution/#linux" y ejecutarlo NO COMO
	    ROOT sino como el usuario con el que queremos usar anaconda.

        2. Elegir el nombre que queramos para el entorno ("planning_with_subgoals" por defecto):
	    solo hay que sustituir la primera línea del archivo "environment.yml" por "name: NOMBRE_ELEGIDO"

	3. Ejecutar el comando "conda env create -f environment.yml"

	    Nota: si se cambia de sistema operativo, es posible que surjan problemas con las dependencias.
	    Yo he realizado todos estos pasos en Ubuntu 18.04 LTS.

