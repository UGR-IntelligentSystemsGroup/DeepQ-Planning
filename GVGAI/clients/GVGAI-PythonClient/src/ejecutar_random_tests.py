# Este archivo va guardando en un archivo el número de acciones (de cada iteración) usadas por un agente
# que escoge las gemas al azar sobre cada uno de los 10 niveles de test

import re
import subprocess
import os 

# SUPONGO QUE COMPETITIONPARAMETERS.PY YA ESTÁN PREPARADOS PARA TRAINING!

path_niveles_juego = '../../../examples/gridphysics/' # Carpeta de donde se cargan los niveles del juego
path_niveles_nuevos = 'NivelesNuevos/' # Carpeta que contiene los 10 nuevos niveles

archivo_output = "num_actions_levels.txt" # Archivo donde se escriben las acciones

id_primer_nivel_nuevo = 5 # Id del primer nivel de test nuevo a probar
id_ultimo_nivel_nuevo = 14 # Id del último nivel de test nuevo a probar

# Voy probando los nuevos niveles de test. En cada iteración, cojo tres nuevos niveles

id_nivel_act = id_primer_nivel_nuevo

try:
	while id_nivel_act <= id_ultimo_nivel_nuevo:
		id_nivel_1 = id_nivel_act
		id_nivel_2 = id_nivel_act+1
		id_nivel_3 = id_nivel_act+2

		# Si ya no hay más niveles que coger, cojo el último que hay
		if id_nivel_2 > id_ultimo_nivel_nuevo:
			id_nivel_2 = id_nivel_1

		if id_nivel_3 > id_ultimo_nivel_nuevo:
			id_nivel_3 = id_nivel_1

		nombre_nuevo_nivel_1 = "boulderdash_lvl{}.txt".format(id_nivel_1)
		nombre_nuevo_nivel_2 = "boulderdash_lvl{}.txt".format(id_nivel_2)
		nombre_nuevo_nivel_3 = "boulderdash_lvl{}.txt".format(id_nivel_3)

		# Elimino los 3 niveles actuales de training en la carpeta del juego
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl0.txt", shell=True)
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl1.txt", shell=True)
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl2.txt", shell=True)

		# Copio los tres siguientes niveles nuevos a la carpeta y les cambio el nombre
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_1 + " " + path_niveles_juego + "boulderdash_lvl0.txt", shell=True)
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_2 + " " + path_niveles_juego + "boulderdash_lvl1.txt", shell=True)
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_3 + " " + path_niveles_juego + "boulderdash_lvl2.txt", shell=True)

		# Escribo en el archivo test_output.txt los niveles de test que voy a probar
		with open(archivo_output, 'a') as file:
			file.write("\n\n\n--------------------------------\n")
			file.write("-   Nivel {}, nivel {} y nivel {}   -\n".format(id_nivel_1, id_nivel_2, id_nivel_3))
			file.write("--------------------------------\n\n\n")

		# Execute training phase
		subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

		# Kill java process so that the memory doesn't fill
		subprocess.call("killall java", shell=True)

		id_nivel_act += 3 # Cojo los siguientes tres niveles

except:
	print("~ Se ha producido una excepción ~")