# Script used to train models and test them on the five validation levels.

import re
import subprocess
import os
import glob
import random

# <Model Hyperparameters>
# This script trains and validates one model per each different combination of
# these hyperparameters

# Architecture
# First conv layer
l1_num_filt = [32]
l1_filter_structure = [ [[5,5],[1,1],"VALID"],  [[4,4],[1,1],"VALID"]]

# Second conv layer
l2_num_filt = [64]
l2_filter_structure = [ [[4,4],[1,1],"VALID"],  [[3,3],[1,1],"VALID"]]

# Third conv layer
l3_num_filt = [128] 
l3_filter_structure = [ [[3,3],[1,1],"VALID"] ]

# Third conv layer
l4_num_filt = [128]
l4_filter_structure = [ [[3,3],[1,1],"VALID"] ]

# Don't use max pooling
"""
max_pool_size = [[2, 2]]
max_pool_str = [[1, 1]]
"""

# A single fc layer works better!
fc_num_unis = [[64,1], [128,1]] # Number of units of the first and second fully-connected layers

# Training params
num_its = [7500] # Number of iterations for training
alfa = [0.005] # Learning rate
dropout = [0.0] # Dropout value
batch_size = [32] # Batch size

# Extra params
games_to_play = ['BoulderDash', 'IceAndFire', 'Catapults']
datasets_sizes_for_training = [20] # For each size, a different model is trained and tested on this number of levels
repetitions_per_model = 1 # Each model is trained this number of times

# <Script variables>

# > Variables for each game

# BoulderDash
lvs_path_boulderdash = "NivelesAllGames/Niveles_BoulderDash/Train_Val/" # Folder where the levels to extract the datasets from are saved
game_id_boulderdash = "11" # Value of the game_id variable in the oneClickRunFromPythonClient.sh script
# Names of the test levels files (lvs 3-4)
test_lvs_boulderdash = ('boulderdash_lvl3.txt', 'boulderdash_lvl4.txt')

# IceAndFire
lvs_path_iceandfire = "NivelesAllGames/Niveles_IceAndFire/Train_Val/"
game_id_iceandfire = "43"
test_lvs_iceandfire = ('iceandfire_lvl3.txt', 'iceandfire_lvl4.txt')

# Catapults
lvs_path_catapults = "NivelesAllGames/Niveles_Catapults/Train_Val/"
game_id_catapults = "16"
test_lvs_catapults = ('catapults_lvl3.txt', 'catapults_lvl4.txt')

test_lvs_directory = "../../../examples/gridphysics/" # Path where the test levels are located


# ----- Execution -----

# Save the hyperparameters for each different model in a list
models_params = [ (a,b,c,d,e,f,g,h,i,j,k,l,m,n,o)
					for a in l1_num_filt for b in l1_filter_structure for c in l2_num_filt for d in l2_filter_structure \
 					for e in l3_num_filt for f in l3_filter_structure for g in l3_num_filt for h in l3_filter_structure \
 					for i in fc_num_unis for j in num_its for k in alfa for l in dropout for m in batch_size \
 					for n in games_to_play for o in datasets_sizes_for_training]

try:
	# Iterate over the different models
	for curr_model_params in models_params:
		# <Current model hyperparameters>
		curr_l1_num_filt = curr_model_params[0]
		curr_l1_filter_structure = curr_model_params[1]
		curr_l2_num_filt = curr_model_params[2]
		curr_l2_filter_structure = curr_model_params[3]
		curr_l3_num_filt = curr_model_params[4]
		curr_l3_filter_structure = curr_model_params[5]
		curr_l4_num_filt = curr_model_params[6]
		curr_l4_filter_structure = curr_model_params[7]
		curr_fc_num_unis = curr_model_params[8]
		curr_num_its = curr_model_params[9]
		curr_alfa = curr_model_params[10]
		curr_dropout = curr_model_params[11]
		curr_batch_size = curr_model_params[12]
		curr_game = curr_model_params[13]
		dataset_size_for_training = curr_model_params[14]

		# Variables that depend on the game being played
		if curr_game == 'BoulderDash':
			curr_lvs_path = lvs_path_boulderdash
			curr_game_id = game_id_boulderdash
			curr_test_lvs = test_lvs_boulderdash
		elif curr_game == 'IceAndFire':
			curr_lvs_path = lvs_path_iceandfire
			curr_game_id = game_id_iceandfire
			curr_test_lvs = test_lvs_iceandfire
		else: # Catapults
			curr_lvs_path = lvs_path_catapults
			curr_game_id = game_id_catapults
			curr_test_lvs = test_lvs_catapults

		# <Change Agent.py>

		# Load file in memory
		with open('MyAgent/Agent.py', 'r') as file:
			agent_file = file.read()

		# Change model params
		agent_file = re.sub(r'self.l1_num_filt=.*', 'self.l1_num_filt={}'.format(curr_l1_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l1_window=.*', 'self.l1_window={}'.format(curr_l1_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l1_strides=.*', 'self.l1_strides={}'.format(curr_l1_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l1_padding_type=.*', 'self.l1_padding_type="{}"'.format(curr_l1_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l2_num_filt=.*', 'self.l2_num_filt={}'.format(curr_l2_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l2_window=.*', 'self.l2_window={}'.format(curr_l2_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l2_strides=.*', 'self.l2_strides={}'.format(curr_l2_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l2_padding_type=.*', 'self.l2_padding_type="{}"'.format(curr_l2_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l3_num_filt=.*', 'self.l3_num_filt={}'.format(curr_l3_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l3_window=.*', 'self.l3_window={}'.format(curr_l3_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l3_strides=.*', 'self.l3_strides={}'.format(curr_l3_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l3_padding_type=.*', 'self.l3_padding_type="{}"'.format(curr_l3_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l4_num_filt=.*', 'self.l4_num_filt={}'.format(curr_l4_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l4_window=.*', 'self.l4_window={}'.format(curr_l4_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l4_strides=.*', 'self.l4_strides={}'.format(curr_l4_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l4_padding_type=.*', 'self.l4_padding_type="{}"'.format(curr_l4_filter_structure[2]), agent_file, count=1)


		agent_file = re.sub(r'self.fc_num_unis=.*', 'self.fc_num_unis={}'.format(curr_fc_num_unis), agent_file, count=1)
		agent_file = re.sub(r'self.learning_rate=.*', 'self.learning_rate={}'.format(curr_alfa), agent_file, count=1)
		agent_file = re.sub(r'self.dropout_prob=.*', 'self.dropout_prob={}'.format(curr_dropout), agent_file, count=1)
		agent_file = re.sub(r'self.num_train_its=.*', 'self.num_train_its={}'.format(curr_num_its), agent_file, count=1)
		agent_file = re.sub(r'self.batch_size=.*', 'self.batch_size={}'.format(curr_batch_size), agent_file, count=1)
		
		# Change other variables
		agent_file = re.sub(r'self.game_playing=.*', 'self.game_playing="{}"'.format(curr_game), agent_file, count=1)
		agent_file = re.sub(r'self.dataset_size_for_training=.*', 'self.dataset_size_for_training={}'.format(dataset_size_for_training), agent_file, count=1)

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
		for curr_rep in range(repetitions_per_model):

			# <Create the model name using the hyperparameters values>

			"""curr_model_name = "DQN_BN_conv1-{},{},{},{}_conv2-{},{},{},{}_conv3-{},{},{},{}_conv4-{},{},{},{}_fc-{}_{}_its-{}_alfa-{}_dropout-{}_batch-{}_{}_{}". \
							format(curr_l1_num_filt, curr_l1_filter_structure[0][0], curr_l1_filter_structure[1][0], curr_l1_filter_structure[2], \
							curr_l2_num_filt, curr_l2_filter_structure[0][0], curr_l2_filter_structure[1][0], curr_l2_filter_structure[2], \
							curr_l3_num_filt, curr_l3_filter_structure[0][0], curr_l3_filter_structure[1][0], curr_l3_filter_structure[2], \
							curr_l4_num_filt, curr_l4_filter_structure[0][0], curr_l4_filter_structure[1][0], curr_l4_filter_structure[2], \
							curr_fc_num_unis[0], curr_fc_num_unis[1], \
							curr_num_its, curr_alfa, curr_dropout, curr_batch_size, curr_game, curr_rep)"""

			curr_model_name = "DQN_prueba_convergencia_training_{}_{}_{}_{}".format(curr_num_its, curr_alfa, curr_batch_size, curr_game)

			print("\n\nCurrent model: {} - Current repetition: {}\n".format(curr_model_name, curr_rep))

			# <Change Agent.py>	

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Change the model name
			agent_file = re.sub(r'self.network_name=.*', 'self.network_name="{}"'.format(curr_model_name), agent_file, count=1)

			# Save file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)


			# ------ TRAINING ------


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


			# ------ VALIDATION ------


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

			# <Select the five validation levels to use>

			# Get all the training/validation levels
			all_levels = glob.glob(curr_lvs_path + "*")

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

			# <Validate the model on a different pair of val levels each time>
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

				print("\n> Validating the model on level(s):", curr_val_levels)
				subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

				# <Kill java process so that the memory doesn't fill>
				subprocess.call("killall java 2> /dev/null", shell=True)

except Exception as e:
	print(">> Exception!!")
	print(e)
finally:
	print(">> ejecutar_prueba.py finished!!")

	# Shutdown the computer
	# subprocess.call("poweroff", shell=True)


					