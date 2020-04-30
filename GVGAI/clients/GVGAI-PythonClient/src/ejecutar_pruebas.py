# -*- coding: utf-8 -*-

import re
import subprocess
import os


try: # Capture any exception to make sure the computer always shuts down

	num_rep = 1 # Number of times the model is trained and tested

	for curr_rep in range(num_rep):

		# ----- TRAINING -----


		training_time = "100*60*MILLIS_IN_MIN" # 100 horas de entrenamiento como máximo



		# < Model Hyperparameters >

		# Ej name: DQN_alfa-0.005_dropout-0.4_batch-16_its-7500_1

		model_id = curr_rep # To differentiate different tests with the same model hyperparameters

		num_its = 500 # Number of iterations for training
		alfa = 0.005 # Learning rate
		dropout = 0.2 # Dropout value
		batch_size = 16 # Batch size



		# Create model name using the hyperparameters values
		network_name = '"DQN_alfa-{}_dropout-{}_batch-{}_its-{}_{}"'.format(alfa, dropout, batch_size, num_its, model_id)

		print("\n\n~ Empezado el entrenamiento de {} ~\n\n".format(network_name))

		# Change Agent.py file

		# Load file in memory
		with open('MyAgent/Agent.py', 'r') as file:
			agent_file = file.read()

		# Change model name
		agent_file = re.sub(r'self.network_name=\".*\"', "self.network_name=" + network_name, agent_file, count=1)

		# Change execution mode to train
		agent_file = re.sub(r'self.EXECUTION_MODE=\".*\"', 'self.EXECUTION_MODE="train"', agent_file, count=1)

		# Change hyperparameters values

		agent_file = re.sub(r'self.learning_rate=.*', 'self.learning_rate={}'.format(alfa), agent_file, count=1)
		agent_file = re.sub(r'self.dropout_prob=.*', 'self.dropout_prob={}'.format(dropout), agent_file, count=1)
		agent_file = re.sub(r'self.num_train_its=.*', 'self.num_train_its={}'.format(num_its), agent_file, count=1)
		agent_file = re.sub(r'self.batch_size=.*', 'self.batch_size={}'.format(batch_size), agent_file, count=1)

		# Write file
		with open('MyAgent/Agent.py', 'w') as file:
			file.write(agent_file)

		# Change CompetitionParameters.py file

		# Load file in memory
		with open('utils/CompetitionParameters.py', 'r') as file:
			comp_param_file = file.read()

		# Change learning time to training time
		comp_param_file = re.sub(r'TOTAL_LEARNING_TIME=.*', "TOTAL_LEARNING_TIME=" + training_time, comp_param_file, count=1)

		# Write file
		with open('utils/CompetitionParameters.py', 'w') as file:
			file.write(comp_param_file)


		# Execute the training with the current hyperparameters and wait until it finishes


		subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)



		# ----- TEST -----



		print("\n\n~ Empezado la validación de {} ~\n\n".format(network_name))

		test_time = "0.1*MILLIS_IN_MIN" # Nada de entrenamiento

		# Change Agent.py file

		# Load file in memory
		with open('MyAgent/Agent.py', 'r') as file:
			agent_file = file.read()

		# Change execution mode to test
		agent_file = re.sub(r'self.EXECUTION_MODE=\".*\"', 'self.EXECUTION_MODE="test"', agent_file, count=1)

		# Write file
		with open('MyAgent/Agent.py', 'w') as file:
			file.write(agent_file)

		# Change CompetitionParameters.py file

		# Load file in memory
		with open('utils/CompetitionParameters.py', 'r') as file:
			comp_param_file = file.read()

		# Change learning time to test time
		comp_param_file = re.sub(r'TOTAL_LEARNING_TIME=.*', "TOTAL_LEARNING_TIME=" + test_time, comp_param_file, count=1)

		# Write file
		with open('utils/CompetitionParameters.py', 'w') as file:
			file.write(comp_param_file)


		# Execute the validation phase with the saved models

		dataset_sizes = [500, 1000, 2500, 5000, 7500, 10000]

		for dataset_size in dataset_sizes:
			# Change dataset size of model to load to "dataset_size" in Agent.py

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Change execution mode to test
			agent_file = re.sub(r'self.num_it_model=.*', 'self.num_it_model={}'.format(dataset_size), agent_file, count=1)

			# Write file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)


			# Execute test phase for current model and dataset size and save the results to disk
			subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

			# Kill java process so that the memory doesn't fill
			subprocess.call("killall java", shell=True)

except:
	print("SE HA PRODUCIDO UNA EXCEPCIÓN!!!")

	subprocess.call("touch SE_HA_PRODUCIDO_UNA_EXCEPCION", shell=True)
finally:
	# Shutdown Computer regardless of exceptions - NEEDS ROOT PRIVILEGES!!
	# os.system("shutdown -h +1") # Shutdown in 1 minute to make it possible to halt it
