# Script used to train models and test them on the five validation levels.

import re
import subprocess
import os
import glob
import random
import sys

"""
Creo que puede que la TargetNetwork no esté recibiendo los pesos de la DQNetwork.
Puedo comprobarlo simplemente eliminando la target network y prediciendo con la DQNetwork.


En los niveles malos de test, veo que se predicen muchos valores cercanos a 0.5 o cercanos
a 0.

Comparación ejecuciones Normal y Q_target network que no se actualizan sus pesos ->
son idénticas hasta 15000 iteraciones o así, donde la ejecución con Q_target network
estática empieza a oscilar mucho su pérdida (no converge a 0).

Sin target network no funciona el entrenamiento (siempre predice un Q_value cercano a 0).
Esto sucede también con la función de pérdida L2!!

Al hacer pruebas con L2 norm loss y Q-target estático, al final la pérdida termina convergiendo
a 0, pero tarda mucho.

Pruebo a aumentar el learning rate. -> alfa = 0.01: NO FUNCIONA (siempre predice Q-value de 0)
Al probar a disminuir el learning rate (alfa=0.001) -> <<<<<La pérdida sí converge a 0!!!!!>>>>>
Al probar alfa=0.0005 -> La pérdida converge a 0 super rápido!!!!
Al probar alfa=0.0001 -> La pérdida converge a 0 un poco más rápido que en el caso anterior.
Al probar alfa=0.00005 -> La pérdida converge un poco más lenta a 0.

>>> Mejor valor de alfa=0.0001
"""






# <Execution mode of the script>
# "validation" -> trains and validates on 5 levels not used for training
# "test" -> trains and tests on the 5 test levels
script_execution_mode = "test" # <<<CAMBIAR>>>

# <Goal Selection Mode>
# "best" -> use the trained model to select the best subgoal at each state
# "random" -> select subgoals randomly. This corresponds to the Random Model
# goal_selection_mode = "best"

# <Model Hyperparameters>
# This script trains and validates one model per each different combination of
# these hyperparameters

# Architecture
# First conv layer
vs_l1_num_filt = [32]
vs_l1_filter_structure = [ [[4,4],[1,1],"VALID"] ]

# Second conv layer
vs_l2_num_filt = [32]
vs_l2_filter_structure = [ [[4,4],[1,1],"VALID"] ]

# Third conv layer
vs_l3_num_filt = [64]
vs_l3_filter_structure = [ [[4,4],[1,1],"VALID"] ]

# Third conv layer
vs_l4_num_filt = [128]
vs_l4_filter_structure = [ [[4,4],[1,1],"VALID"] ]

# A single fc layer works better!
vs_fc_num_unis = [[32, 1]] # Number of units of the first and second fully-connected layers

# Training params
vs_num_its = [10000] # Number of iterations for training #7500
vs_tau=[250] # Update period of the target network
vs_alfa = [0.00005] # 0.005 # Learning rate # 0.01 is too much
vs_dropout = [0.0] # Dropout value
vs_batch_size = [16] # 16 works better than 32 for test. For training loss, 32 works better than 16.

# Extra params
# games_to_play = ['BoulderDash', 'IceAndFire', 'Catapults']
games_to_play = ['BoulderDash']
# For each size, a different model is trained and tested on this number of levels
datasets_sizes_for_training_BoulderDash = [1] # 20 # 25
datasets_sizes_for_training_IceAndFire = [45] # 50
datasets_sizes_for_training_Catapults = [95] # 100
repetitions_per_model = 1 # 30 # Each model is trained this number of times

# <Script variables>

# > Variables for each game

# BoulderDash
lvs_path_boulderdash_train_val = "NivelesAllGames/Niveles_BoulderDash/Train_Val/" # Folder where the training and validation levels are saved
lvs_path_boulderdash_test = "NivelesAllGames/Niveles_BoulderDash/Test/" # Folder where the test levels are saved
game_id_boulderdash = "11" # Value of the game_id variable in the oneClickRunFromPythonClient.sh script
# Names of the test levels files (lvs 3-4)
test_lvs_boulderdash = ('boulderdash_lvl3.txt', 'boulderdash_lvl4.txt')

# IceAndFire
lvs_path_iceandfire_train_val = "NivelesAllGames/Niveles_IceAndFire/Train_Val/"
lvs_path_iceandfire_test = "NivelesAllGames/Niveles_IceAndFire/Test/"
game_id_iceandfire = "43"
test_lvs_iceandfire = ('iceandfire_lvl3.txt', 'iceandfire_lvl4.txt')

# Catapults
lvs_path_catapults_train_val = "NivelesAllGames/Niveles_Catapults/Train_Val/"
lvs_path_catapults_test = "NivelesAllGames/Niveles_Catapults/Test/"
game_id_catapults = "16"
test_lvs_catapults = ('catapults_lvl3.txt', 'catapults_lvl4.txt')

test_lvs_directory = "../../../examples/gridphysics/" # Path where the test levels are located


# ----- Execution -----

# Save the hyperparameters for each different model in a list
models_params = [ (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p_b) if o == 'BoulderDash' else 
				  (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p_i) if o == 'IceAndFire' else
				  (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p_c)
					for a in vs_l1_num_filt for b in vs_l1_filter_structure for c in vs_l2_num_filt for d in vs_l2_filter_structure \
 					for e in vs_l3_num_filt for f in vs_l3_filter_structure for g in vs_l4_num_filt for h in vs_l4_filter_structure \
 					for i in vs_fc_num_unis for j in vs_num_its for k in vs_alfa for l in vs_dropout for m in vs_batch_size \
					for n in vs_tau for o in games_to_play 
 					for p_b in datasets_sizes_for_training_BoulderDash for p_i in datasets_sizes_for_training_IceAndFire
 					for p_c in datasets_sizes_for_training_Catapults]

try:
	# Iterate over the different models
	for curr_model_params in models_params:
		# <Current model hyperparameters>
		curr_vs_l1_num_filt = curr_model_params[0]
		curr_vs_l1_filter_structure = curr_model_params[1]
		curr_vs_l2_num_filt = curr_model_params[2]
		curr_vs_l2_filter_structure = curr_model_params[3]
		curr_vs_l3_num_filt = curr_model_params[4]
		curr_vs_l3_filter_structure = curr_model_params[5]
		curr_vs_l4_num_filt = curr_model_params[6]
		curr_vs_l4_filter_structure = curr_model_params[7]
		curr_vs_fc_num_unis = curr_model_params[8]
		curr_vs_num_its = curr_model_params[9]
		curr_vs_alfa = curr_model_params[10]
		curr_vs_dropout = curr_model_params[11]
		curr_vs_batch_size = curr_model_params[12]
		curr_vs_tau = curr_model_params[13]
		curr_game = curr_model_params[14]
		dataset_size_for_training = curr_model_params[15]

		# Variables that depend on the game being played
		if curr_game == 'BoulderDash':
			curr_lvs_path_train_val = lvs_path_boulderdash_train_val
			curr_lvs_path_test = lvs_path_boulderdash_test
			curr_game_id = game_id_boulderdash
			curr_test_lvs = test_lvs_boulderdash
		elif curr_game == 'IceAndFire':
			curr_lvs_path_train_val = lvs_path_iceandfire_train_val
			curr_lvs_path_test = lvs_path_iceandfire_test
			curr_game_id = game_id_iceandfire
			curr_test_lvs = test_lvs_iceandfire
		else: # Catapults
			curr_lvs_path_train_val = lvs_path_catapults_train_val
			curr_lvs_path_test = lvs_path_catapults_test
			curr_game_id = game_id_catapults
			curr_test_lvs = test_lvs_catapults

		# <Change Agent.py>

		# Load file in memory
		with open('MyAgent/Agent.py', 'r') as file:
			agent_file = file.read()

		# Change model params
		agent_file = re.sub(r'self.vs_l1_num_filt=.*', 'self.vs_l1_num_filt={}'.format(curr_vs_l1_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l1_window=.*', 'self.vs_l1_window={}'.format(curr_vs_l1_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l1_strides=.*', 'self.vs_l1_strides={}'.format(curr_vs_l1_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l1_padding_type=.*', 'self.vs_l1_padding_type="{}"'.format(curr_vs_l1_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.vs_l2_num_filt=.*', 'self.vs_l2_num_filt={}'.format(curr_vs_l2_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l2_window=.*', 'self.vs_l2_window={}'.format(curr_vs_l2_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l2_strides=.*', 'self.vs_l2_strides={}'.format(curr_vs_l2_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l2_padding_type=.*', 'self.vs_l2_padding_type="{}"'.format(curr_vs_l2_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.vs_l3_num_filt=.*', 'self.vs_l3_num_filt={}'.format(curr_vs_l3_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l3_window=.*', 'self.vs_l3_window={}'.format(curr_vs_l3_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l3_strides=.*', 'self.vs_l3_strides={}'.format(curr_vs_l3_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l3_padding_type=.*', 'self.vs_l3_padding_type="{}"'.format(curr_vs_l3_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.vs_l4_num_filt=.*', 'self.vs_l4_num_filt={}'.format(curr_vs_l4_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l4_window=.*', 'self.vs_l4_window={}'.format(curr_vs_l4_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l4_strides=.*', 'self.vs_l4_strides={}'.format(curr_vs_l4_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.vs_l4_padding_type=.*', 'self.vs_l4_padding_type="{}"'.format(curr_vs_l4_filter_structure[2]), agent_file, count=1)


		agent_file = re.sub(r'self.vs_fc_num_unis=.*', 'self.vs_fc_num_unis={}'.format(curr_vs_fc_num_unis), agent_file, count=1)
		agent_file = re.sub(r'self.vs_learning_rate=.*', 'self.vs_learning_rate={}'.format(curr_vs_alfa), agent_file, count=1)
		agent_file = re.sub(r'self.vs_dropout_prob=.*', 'self.vs_dropout_prob={}'.format(curr_vs_dropout), agent_file, count=1)
		agent_file = re.sub(r'self.vs_num_train_its=.*', 'self.vs_num_train_its={}'.format(curr_vs_num_its), agent_file, count=1)
		agent_file = re.sub(r'self.vs_batch_size=.*', 'self.vs_batch_size={}'.format(curr_vs_batch_size), agent_file, count=1)
		agent_file = re.sub(r'self.vs_max_tau=.*', 'self.vs_max_tau={}'.format(curr_vs_tau), agent_file, count=1)

		# Change other variables
		agent_file = re.sub(r'self.game_playing=.*', 'self.game_playing="{}"'.format(curr_game), agent_file, count=1)
		agent_file = re.sub(r'self.dataset_size_for_training=.*', 'self.dataset_size_for_training={}'.format(dataset_size_for_training), agent_file, count=1)

		# Change goal_selection_mode
		# agent_file = re.sub(r'self.goal_selection_mode=.*', 'self.goal_selection_mode="{}"'.format(goal_selection_mode), agent_file, count=1)

		# Save file
		with open('MyAgent/Agent.py', 'w') as file:
			file.write(agent_file)

		# <Change oneClickRunFromPythonClient.sh>

		# Load file in memory
		with open('oneclickRunFromPythonClient.sh', 'r') as file:
			oneclickrun_file = file.read()

		# Set game_id
		oneclickrun_file = re.sub(r'game_id=.+', "game_id={}".format(curr_game_id), oneclickrun_file, count=1)

		# Save file
		with open('oneclickRunFromPythonClient.sh', 'w') as file:
			file.write(oneclickrun_file)


		# <Repeat each execution (train + val) the number of times given by "repetitions_per_model">
		# If the goal selection mode is random, skip the training part
		for curr_rep in range(repetitions_per_model):

			# <Create the model name using the hyperparameters values>

			if script_execution_mode == "validation":
				curr_vs_model_name = "DQNValidSubgoals_3_conv1-{},{},{},{}_conv2-{},{},{},{}_conv3-{},{},{},{}_conv4-{},{},{},{}_fc-{}_{}_its-{}_alfa-{}_dropout-{}_batch-{}_tau-{}_{}_{}". \
								format(curr_vs_l1_num_filt, curr_vs_l1_filter_structure[0][0], curr_vs_l1_filter_structure[1][0], curr_vs_l1_filter_structure[2], \
								curr_vs_l2_num_filt, curr_vs_l2_filter_structure[0][0], curr_vs_l2_filter_structure[1][0], curr_vs_l2_filter_structure[2], \
								curr_vs_l3_num_filt, curr_vs_l3_filter_structure[0][0], curr_vs_l3_filter_structure[1][0], curr_vs_l3_filter_structure[2], \
								curr_vs_l4_num_filt, curr_vs_l4_filter_structure[0][0], curr_vs_l4_filter_structure[1][0], curr_vs_l4_filter_structure[2], \
								curr_vs_fc_num_unis[0], curr_vs_fc_num_unis[1], \
								curr_vs_num_its, curr_vs_alfa, curr_vs_dropout, curr_vs_batch_size, curr_vs_tau, curr_game, curr_rep)
			else:
				curr_vs_model_name = "DQNValidSubgoals_prueba_overfitting_L2_loss_Q_target_estatico_lvl-{}_tau-{}_alfa-{}_{}_{}".format(curr_vs_num_its,
				 curr_vs_tau, curr_vs_alfa, curr_game, curr_rep)

			print("\n\nCurrent model: {} - Current repetition: {}\n".format(curr_vs_model_name, curr_rep))

			# <Change Agent.py>	

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Change the model name
			agent_file = re.sub(r'self.vs_network_name=.*', 'self.vs_network_name="{}"'.format(curr_vs_model_name), agent_file, count=1)

			# Save file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)


			# ------ TRAINING ------

			# Skip training if we are testing the random model
			# if goal_selection_mode == "best":

			# <Change Agent.py>

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Change execution mode
			agent_file = re.sub(r'self.EXECUTION_MODE=.*', 'self.EXECUTION_MODE="train"', agent_file, count=1)

			# Save file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)

			# <Change CompetitionParameters.py>

			# Load file in memory
			with open('utils/CompetitionParameters.py', 'r') as file:
				comp_param_file = file.read()

			# Change learning time to training time
			comp_param_file = re.sub(r'TOTAL_LEARNING_TIME=.*', "TOTAL_LEARNING_TIME=100*60*MILLIS_IN_MIN", comp_param_file, count=1)

			# Save file
			with open('utils/CompetitionParameters.py', 'w') as file:
				file.write(comp_param_file)

			# <Execute the training with the current hyperparameters and wait until it finishes>

			print("\n> Starting the training of the current model")
			subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)


			# ------ VALIDATION / TEST ------


			# <Change Agent.py>

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Change execution mode
			agent_file = re.sub(r'self.EXECUTION_MODE=.*', 'self.EXECUTION_MODE="test"', agent_file, count=1)
			# Change dataset size
			agent_file = re.sub(r'self.dataset_size_model=.*', 'self.dataset_size_model={}'.format(dataset_size_for_training), agent_file, count=1)

			# Save file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)

			# <Change CompetitionParameters.py>

			# Load file in memory
			with open('utils/CompetitionParameters.py', 'r') as file:
				comp_param_file = file.read()

			# Change learning time to test time
			comp_param_file = re.sub(r'TOTAL_LEARNING_TIME=.*', "TOTAL_LEARNING_TIME=1", comp_param_file, count=1)

			# Save file
			with open('utils/CompetitionParameters.py', 'w') as file:
				file.write(comp_param_file)

			# <Select the five validation or test levels to use, depending on the
			# script_execution_mode>

			# Select five validation levels
			if script_execution_mode == "validation":
				# Get all the training/validation levels
				all_levels = glob.glob(curr_lvs_path_train_val + "*")

				# Get the datasets used to train the model
				with open('loaded_datasets.txt', 'r') as file:
					train_datasets = file.read().splitlines()

				# The dataset of id 'j' has been collected at lv of id 'j': transform the datasets into their corresponding levels
				# Ids of the train datasets (e.g.: [5, 7, 21])
				train_datasets_ids = [int(re.search(r'[0-9]+.dat', dataset).group(0).rstrip('.dat')) for dataset in train_datasets]

				# Remove the levels used for training
				levels_to_remove = []

				for lv in all_levels:
					# Get lv id
					lv_id = int(re.search(r'lvl[0-9]+', lv).group(0).lstrip('lvl'))

					# If the lv id is in train_datasets_ids, that means that level was used for training:
					# then don't use it for validation
					if lv_id in train_datasets_ids:
						levels_to_remove.append(lv)

				all_levels = [lv for lv in all_levels if lv not in levels_to_remove]

				# Select 5 validation levels among all the possible levels
				val_levels = random.sample(all_levels, k=5)
				print("\n> Validation levels:", val_levels)

			# Use the five test levels
			else:
				# Get all the test levels
				val_levels = glob.glob(curr_lvs_path_test + "*")
				print("\n> Test levels:", val_levels)

			# <Validate the model on a different pair of val/test levels each time>
			for curr_val_levels in [(0,1),(2,3),(4,)]:

				# <Remove the test levels (3-4) of the corresponding game>
				test_levels_current_game = [test_lvs_directory + level_name for level_name in curr_test_lvs]
				
				for level in test_levels_current_game:
					subprocess.call("rm {} 2> /dev/null".format(level), shell=True)

				if len(curr_val_levels) == 1: # Only one validation level to test
					# <Copy the new validation level as the levels 3-4>
					subprocess.call("cp {} {}".format(val_levels[curr_val_levels[0]], test_levels_current_game[0]), shell=True) 
					subprocess.call("cp {} {}".format(val_levels[curr_val_levels[0]], test_levels_current_game[1]), shell=True) 

					# <Change Agent.py>

					# Load file in memory
					with open('MyAgent/Agent.py', 'r') as file:
						agent_file = file.read()

					# Change num_test_levels
					agent_file = re.sub(r'self.num_test_levels=.*', 'self.num_test_levels=1', agent_file, count=1)

					# Save file
					with open('MyAgent/Agent.py', 'w') as file:
						file.write(agent_file)

				else: # Two validation levels to test
					# <Copy the new validation levels as the levels 3-4>
					subprocess.call("cp {} {}".format(val_levels[curr_val_levels[0]], test_levels_current_game[0]), shell=True) 
					subprocess.call("cp {} {}".format(val_levels[curr_val_levels[1]], test_levels_current_game[1]), shell=True) 

					# <Change Agent.py>

					# Load file in memory
					with open('MyAgent/Agent.py', 'r') as file:
						agent_file = file.read()

					# Change num_test_levels
					agent_file = re.sub(r'self.num_test_levels=.*', 'self.num_test_levels=2', agent_file, count=1)

					# Save file
					with open('MyAgent/Agent.py', 'w') as file:
						file.write(agent_file)


				# <Execute the validation on the current validation levels>

				print("\n> Validating/Testing the model on level(s):", curr_val_levels)
				subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

				# <Kill java process so that the memory doesn't fill>
				subprocess.call("killall java 2> /dev/null", shell=True)

except Exception as e:
	print(">> Exception!!")
	print(e)
finally:
	print(">> ejecutar_prueba.py finished!!")

	# Shutdown the computer in a minute
	# subprocess.call("shutdown -t 60", shell=True)


					