# -*- coding: utf-8 -*-

import re
import subprocess
import os

# SUPONGO QUE ĹOS ARCHIVOS AGENT.PY Y COMPETITIONPARAMETERS.PY YA ESTÁN PREPARADOS PARA TEST!

path_niveles_juego = '../../../examples/gridphysics/' # Carpeta de donde se cargan los niveles del juego
path_niveles_nuevos = 'NivelesNuevos/' # Carpeta que contiene los 10 nuevos niveles


nombre_modelos = "DQN_alfa-0.005_dropout-0.4_batch-16_its-5000_" # Hay que añadir al final la id del modelo

id_ini = 13 # Id del primer modelo a testear
id_fin = 27 # Id del último modelo a testear

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