# Executes Agent.py succesively in "create_dataset" mode to create the datasets for all three games.

import re
import subprocess
import os
import glob
import sys

# BoulderDash
lvs_path_boulderdash = "NivelesAllGames/Niveles_BoulderDash/Train_Val/" # Folder where the levels to extract the datasets from are saved
game_playing_boulderdash = "BoulderDash" # Value of the self.game_playing attribute in Agent.py
game_id_boulderdash = "11" # Value of the game_id variable in the oneClickRunFromPythonClient.sh script
# Names of the training levels files (lvs 0-2)
training_lvs_boulderdash = ('boulderdash_lvl0.txt', 'boulderdash_lvl1.txt', 'boulderdash_lvl2.txt')

# IceAndFire
lvs_path_iceandfire = "NivelesAllGames/Niveles_IceAndFire/Train_Val/"
game_playing_iceandfire = "IceAndFire"
game_id_iceandfire = "43"
training_lvs_iceandfire = ('iceandfire_lvl0.txt', 'iceandfire_lvl1.txt', 'iceandfire_lvl2.txt')

# Catapults
lvs_path_catapults = "NivelesAllGames/Niveles_Catapults/Train_Val/"
game_playing_catapults = "Catapults"
game_id_catapults = "16"
training_lvs_catapults = ('catapults_lvl0.txt', 'catapults_lvl1.txt', 'catapults_lvl2.txt')

# Other variables
# games_to_play = ['BoulderDash', 'IceAndFire', 'Catapults']
games_to_play = ['IceAndFire', 'BoulderDash']

training_lvs_directory = "../../../examples/gridphysics/" # Path where the training levels (0-2) are located

try:
	for current_game in games_to_play:

		# <Set the variables that depend on the game being played>
		if current_game == 'BoulderDash':
			lvs_path = lvs_path_boulderdash
			game_playing = game_playing_boulderdash
			game_id = game_id_boulderdash
			training_lvs = training_lvs_boulderdash
		elif current_game == 'IceAndFire':
			lvs_path = lvs_path_iceandfire
			game_playing = game_playing_iceandfire
			game_id = game_id_iceandfire
			training_lvs = training_lvs_iceandfire
		else: # Catapults
			lvs_path = lvs_path_catapults
			game_playing = game_playing_catapults
			game_id = game_id_catapults
			training_lvs = training_lvs_catapults

		# <Set Agent.py and oneClickRunFromPythonClient.sh files for the the game being currently played>

		# Load Agent.py
		with open('MyAgent/Agent.py', 'r') as file:
			agent_file = file.read()

		# Change execution mode
		agent_file = re.sub(r'self.EXECUTION_MODE=.*', 'self.EXECUTION_MODE="create_dataset"', agent_file, count=1)

		# Set game_playing
		agent_file = re.sub(r'self.game_playing=.+', "self.game_playing='{}'".format(game_playing), agent_file, count=1)

		# Save Agent.py
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

		# Load oneClickRunFromPythonClient.sh
		with open('oneclickRunFromPythonClient.sh', 'r') as file:
			oneclickrun_file = file.read()

		# Set game_id
		oneclickrun_file = re.sub(r'game_id=.+', "game_id={}".format(game_id), oneclickrun_file, count=1)

		# Save oneClickRunFromPythonClient.sh
		with open('oneclickRunFromPythonClient.sh', 'w') as file:
			file.write(oneclickrun_file)

		# <Create the datasets for the current game>

		# Get the paths of all the levels to extract the datasets from
		game_lv_files = glob.glob(lvs_path + "*")
		
		# Iterate over each level of the current game
		for curr_game_lv_file in game_lv_files:
			print("\n\n>> Level playing: {}\n".format(curr_game_lv_file))

			# Get the paths of the training levels (0-2) of the current game
			training_lvs_paths = [training_lvs_directory + level_name for level_name in training_lvs]

			# Remove the training levels
			for training_lv in training_lvs_paths:
				subprocess.call("rm {} 2> /dev/null".format(training_lv), shell=True)

			# Copy the new training level as the levels 0-2
			for training_lv in training_lvs_paths:
				subprocess.call("cp {} {}".format(curr_game_lv_file, training_lv), shell=True) 

			# Load Agent.py
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Get the id of the current level. That will be the dataset id.
			curr_dataset_id = int(re.search(r'lvl[0-9]+', curr_game_lv_file).group(0).lstrip('lvl'))

			print("\n\nCurr_dataset_id =", curr_dataset_id)

			# Change dataset id
			agent_file = re.sub(r'id_dataset=.+', "id_dataset={}".format(curr_dataset_id), agent_file, count=1)

			# Save Agent.py
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)

			# Execute Agent.py (create current dataset)
			subprocess.call("bash oneclickRunFromPythonClient.sh", shell=True)

			# Kill java process so that the memory doesn't fill
			subprocess.call("killall java 2> /dev/null", shell=True)

except Exception as e:
	print(">> Exception!!")
	print(e)
finally:
	print(">> All datasets have been created")

	# Shutdown the computer in a minute
	# subprocess.call("shutdown -t 60", shell=True)

