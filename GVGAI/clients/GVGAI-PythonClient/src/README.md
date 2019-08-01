## Iteracion: 1

# Descripción de la iteración

En esta primera iteración se ha implementado un agente que funciona solo en el
nivel 0 del juego de Boulder Dash. El agente tiene un objetivo muy simple: coger
una gema, la cuál ha sido especificada en un archivo concreto que se explicará
un poco más adelante.

El agente todavía no tiene la capacidad de escoger cuál será su objetivo (ni si quiera
de forma ****greedy*). Esto será considerado en las siguientes iteraciones, donde se va
a proponer que el agente sea capaz de completar un nivel. Esto supondrá, entre otras
cosas, añadir funcionalidad para traducir la observación del nivel de forma que sea
leído por el *parser* posteriormente, con el objetivo de encontrar un plan.

Se ha decidido utilizar el planificador **LAMA**, debido a que ha sido capaz de
ofrecer respuesta a un problema tan grande como el que se está intentando abarcar.
Sin embargo, no se ha conseguido que se pueda realizar todo el proceso de *parsing*
en cuestión de segundos, debido a la complejidad del dominio. Hay que considerar que
**LAMA** necesita unas etapas previas de traducción y preprocesado, las cuáles necesitan
un cierto tiempo.

La implementación del agente se encuentra en la ruta GVGAI/clients/GVGAI-PythonClient/src/MyAgent.
El parser se encuentra en GVGAI/clients/GVGAI-PythonClient/src/MyAgent/parser. Este directorio
incluye tanto **LAMA** como el parser escrito en Python que se encarga de llamar a **LAMA**
y de transformar el plan en acciones interpretables posteriormente.

Se han modificado los parametros de competición tanto en el cliente como en el servidor. Se ha
dejado un tiempo de 500s para el metodo `act()`, y se descalifica a partir de los 510s. El tiempo
de inicialización ha pasado a ser 1000s, por si se quieren realizar inicializaciones en el constructor.

## Entorno de pruebas

Las pruebas se han realizado en el sistema operativo Ubuntu-18.04. No se asegura que funcione
del todo en Windows. Posiblemente sea necesario modificar el .bat para establecer los valores
correctos.

## Requisitos previos

Para poder ejecutar el agente y **LAMA**, se necesita lo siguiente:

1. Las utilidades GNU de make, con tal de compilar los programas necesarios.
2. Python2. Se necesita tener instalado en el equipo una versión de Python2 para poder ejecutar
el traductor de **LAMA**. Se recomienda tener instalada la versión 2.7 o la más reciente.
3. Python3. Se necesita tener instalado en el equipo una versión de Python3 para ejecutar el
cliente y el parser. Se recomienda tener instalada la versión 3.7 o la más reciente.

## Instrucciones de uso

Para ejecutar el cliente, simplemente hay que realizar los siguiente:

1. Situarse en el directorio GVGAI/clients/GVGAI-PythonClient/src/MyAgent/parser. Acceder a los
directorios preprocess y search y compilar los fuentes ejecutando `make`. De ser necesario, deberán
crearse los directorios obj/ para los archivos .o.

2. Situarse en el directorio GVGAI/clients/GVGAI-PythonClient/src y ejecutar:

```sh
./oneClickRunFromPythonClient.sh
```

Una vez que se muestre el nivel con el agente en la posición objetivo, se puede interrumpir la
ejecución el programa con `Ctrl+C`. Se recomienda encarecidamente ejecutar `make clean` una vez
terminado para eliminar los archivos generados por **LAMA**, los cuáles ocupan un espacio considerable.
