## Qué es

El archivo **environment.yml** describe un entorno de Anaconda (conda environment). 
Este entorno es el que estoy usando para ejecutar el código. Usa python 3.5 e incluye los paquetes necesarios
(hasta la fecha) como *TensorFlow*.

## Uso

Para importar este archivo y crear el entorno de conda, hay que seguir los siguientes pasos:

1. Instalar Anaconda (preferiblemente la última versión): hay que descargar el
script bash (en el caso de Linux) de la 
[siguiente página](https://www.anaconda.com/distribution/#linux) y ejecutarlo
**NO COMO ROOT** sino como el usuario con el que queremos usar anaconda.

2. Elegir el nombre que queramos para el entorno (`planning_with_subgoals` por defecto).
Para ello, solo hay que sustituir la primera línea del archivo `environment.yml` por
`name: NOMBRE_ELEGIDO`.

3. Ejecutar el comando `conda env create -f environment.yml`.

	    Nota: si se cambia de sistema operativo, es posible que surjan problemas con las dependencias.
	    Yo he realizado todos estos pasos en Ubuntu 18.04 LTS.
	    
4. Una vez añadido el entorno, es necesario activarlo. Para ello hay que ejecutar
`conda activate nombre_entorno` en Linux y `activate nombre_entorno` en Windows.
Tras ser activado, nos debería aparecer a la izquierda de nuestro nombre en el
terminal `(nombre_entorno)`. Si ahora ejecutamos `python`, no se ejecutará la versión
del sistema sino la del entorno, de forma que tendremos acceso a todos los módulos
instalados en el entorno.
	   
5. Si queremos desactivar el entorno, hay que ejecutar la orden `conda deactivate nombre_entorno`
o `deactivate nombre_entorno`, según nuestro SO. Tras haberlo hecho, si ejecutamos `python`
de nuevo, volveremos a tener acceso a la versión de Python del sistema, no la del entorno.



