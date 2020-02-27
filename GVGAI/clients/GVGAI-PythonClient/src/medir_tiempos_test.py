# Este script va ejecutando el modelo DQP entrenado sobre los nuevos niveles (como niveles de test) y va guardando
# el número de acciones que tardan en completarse

import re
import subprocess
import os 

# SUPONGO QUE COMPETITIONPARAMETERS.PY YA ESTÁN PREPARADOS PARA TRAINING!

path_niveles_juego = '../../../examples/gridphysics/' # Carpeta de donde se cargan los niveles del juego
path_niveles_nuevos = 'NivelesNuevos/' # Carpeta que contiene los 10 nuevos niveles

archivo_output = "plan_times_test.txt" # Archivo donde se escriben los tiempos

# ME HE QUEDADO SOLO CON 6 NIVELES!!!
id_primer_nivel_nuevo = 5 # Id del primer nivel de test nuevo a probar
id_ultimo_nivel_nuevo = 10 # Id del último nivel de test nuevo a probar

# Número de repeticiones para cada pareja de niveles de test
num_rep = 10

# Voy probando los nuevos niveles de test. En cada iteración, cojo dos nuevos niveles

id_nivel_act = id_primer_nivel_nuevo

try:
	while id_nivel_act <= id_ultimo_nivel_nuevo:
		id_nivel_1 = id_nivel_act
		id_nivel_2 = id_nivel_act+1

		# Si ya no hay más niveles que coger, cojo el último que hay
		if id_nivel_2 > id_ultimo_nivel_nuevo:
			id_nivel_2 = id_nivel_1

		nombre_nuevo_nivel_1 = "boulderdash_lvl{}.txt".format(id_nivel_1)
		nombre_nuevo_nivel_2 = "boulderdash_lvl{}.txt".format(id_nivel_2)

		# Elimino los 2 niveles actuales de test en la carpeta del juego
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl3.txt", shell=True)
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl4.txt", shell=True)

		# Copio los dos siguientes niveles nuevos a la carpeta y les cambio el nombre
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_1 + " " + path_niveles_juego + "boulderdash_lvl3.txt", shell=True)
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_2 + " " + path_niveles_juego + "boulderdash_lvl4.txt", shell=True)

		# Escribo en el archivo test_output.txt los niveles de test que voy a probar
		with open(archivo_output, 'a') as file:
			file.write("\n\n--------------{} - {}----------------\n".format(id_nivel_1, id_nivel_2))

		# Repeat number_rep times
		for i in range(num_rep):
			# Execute test phase
			subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

			# Kill java process so that the memory doesn't fill
			subprocess.call("killall java", shell=True)

		id_nivel_act += 2 # Cojo los siguientes dos niveles

except:
	print("~ Se ha producido una excepción ~")