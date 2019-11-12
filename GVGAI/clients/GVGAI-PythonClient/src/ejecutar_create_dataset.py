# Ejecuta sucesivamente create dataset para crear los datasets
# Supongo que el modo ya es create_dataset y que el tiempo de entrenamiento es el correcto!!

import re
import subprocess
import os

id_dataset_ini = 4
id_dataset_fin = 10

try:
	for curr_id in range(id_dataset_ini, id_dataset_fin+1):
		# Load file in memory
		with open('MyAgent/Agent.py', 'r') as file:
			agent_file = file.read()

		# Change dataset id
		agent_file = re.sub(r'id_dataset=.+', "id_dataset={}".format(curr_id), agent_file, count=1)

		# Update file in memory
		with open('MyAgent/Agent.py', 'w') as file:
			file.write(agent_file)

		# Execute agent.py (create current dataset)
		subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)
except:
	pass # Ignore exceptions
finally:
	print(">> Se han creado todos los datasets")

