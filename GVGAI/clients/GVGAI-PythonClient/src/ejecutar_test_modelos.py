# -*- coding: utf-8 -*-

import re
import subprocess
import os

# SUPONGO QUE ĹOS ARCHIVOS AGENT.PY Y COMPETITIONPARAMETERS.PY YA ESTÁN PREPARADOS PARA TEST!

path_niveles_juego = '../../../examples/gridphysics/' # Carpeta de donde se cargan los niveles del juego
path_niveles_nuevos = 'NivelesNuevos/' # Carpeta que contiene los 10 nuevos niveles

id_primer_nivel_nuevo = 5 # Id del primer nivel de test nuevo a probar
id_ultimo_nivel_nuevo = 14 # Id del último nivel de test nuevo a probar

# Voy probando los nuevos niveles de test. En cada iteración, cojo una nueva pareja

id_nivel_act = id_primer_nivel_nuevo

try:
	while id_nivel_act < id_ultimo_nivel_nuevo:
		nombre_nuevo_nivel_1 = "boulderdash_lvl{}.txt".format(id_nivel_act)
		nombre_nuevo_nivel_2 = "boulderdash_lvl{}.txt".format(id_nivel_act+1)

		# Elimino los 2 niveles actuales de test en la carpeta del juego
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl3.txt", shell=True)
		subprocess.call("rm " + path_niveles_juego + "boulderdash_lvl4.txt", shell=True)

		# Copio los dos siguientes niveles nuevos a la carpeta y les cambio el nombre
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_1 + " " + path_niveles_juego + "boulderdash_lvl3.txt", shell=True)
		subprocess.call("cp " + path_niveles_nuevos + nombre_nuevo_nivel_2 + " " + path_niveles_juego + "boulderdash_lvl4.txt", shell=True)

		# Escribo en el archivo test_output.txt los niveles de test que voy a probar
		with open('test_output.txt', 'a') as file:
			file.write("\n\n\n--------------------------------\n")
			file.write("-   Nivel {} y nivel {}   -\n".format(id_nivel_act, id_nivel_act+1))
			file.write("--------------------------------\n\n\n")

		nombre_modelos = "Greedy_alfa-0.002_dropout-0.5_batch-16_its-5000_" # Hay que añadir al final la id del modelo

		id_ini = 1 # Id del primer modelo a testear
		id_fin = 15 # Id del último modelo a testear

		# Prueba todos los modelos entrenados
		for id_act in range(id_ini, id_fin+1):
			# Cambio el archivo Agent.py
			nombre_modelo_act = nombre_modelos + str(id_act)

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Escribo el nombre del modelo actual
			agent_file = re.sub(r'self.network_name=\".*\"', 'self.network_name="{}"'.format(nombre_modelo_act), agent_file, count=1)

			# Write file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)

			dataset_sizes = [500, 1000, 2500, 5000, 7500, 10000]

			# Ejecuto ese modelo para todos los distintos tamaños de datasets que se han guardado
			for dataset_size in dataset_sizes:
				# Change dataset size of model to load to "dataset_size" in Agent.py

				# Load file in memory
				with open('MyAgent/Agent.py', 'r') as file:
					agent_file = file.read()

				# Change dataset size
				agent_file = re.sub(r'self.num_it_model=.*', 'self.num_it_model={}'.format(dataset_size), agent_file, count=1)

				# Write file
				with open('MyAgent/Agent.py', 'w') as file:
					file.write(agent_file)


				# Execute test phase for current model and dataset size and save the results to disk
				subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

				# Kill java process so that the memory doesn't fill
				subprocess.call("killall java", shell=True)


		id_nivel_act += 2 # Cojo la siguiente pareja de niveles

except:
	print("~ Se ha producido una excepción ~")