# Script used to train models and test them on the five validation levels.

import re
import subprocess
import os
import glob
import random
import sys


"""
Mejor arquitectura 3 capas conv. -> DQN_Pruebas_val_conv1-32,4,1,VALID_conv2-64,4,1,VALID_conv3-64,4,1,VALID_conv4--1,4,1,VALID_fc-32_1_its-2500_alfa-0.0001_dropout-0.0_batch-32_tau-10
[32, 64, 64]
Un poco peor que la mejor con 4 capas 

Si se ven las gráficas de entrenamiento, se aprecia
que para BoulderDash tiende a cero pero para Catapults y
IceAndFire no disminuyen más después de un punto!!! (arquitectura
con cinco capas)

He comprobado que se guardan bien los samples (el one_hot_matrix es correcto para
cada estado del juego)

---- Pruebas gráficas entrenamiento

El modelo Full-FC funciona mucho mejor en entrenamiento que usar una CNN!!!!

# Al aumentar el número de niveles de los juegos, aumenta el training error. Parece que al aumentar el número de niveles
# el modelo DQN no puede memorizar!! -> tengo suficientes samples como para crear un modelo muy complejo!!!

# Cuando aplico dropout, las gráficas de entrenamiento son prácticamente iguales (es bueno porque
# no empeora el entrenamiento pero ayuda al test!!)

Los resultados en test de los modelos fully-connected son muy malos!!
Sin embargo, las gráficas de training son muy buenas y el loss converge en los tres juegos.
BoulderDash -> loss de alrededor de 200 al final
Catapults y IceAndFire -> loss alrededor de 3000 al final.

----

>>> Comparación gráficas de entrenamiento modelos FC, conv 4 capas y conv 6 capas:
> BoulderDash: modelo FC -> loss=200 (converge casi perfectamente), 
               modelos conv (4 y 6 capas) -> loss=1000 (convergen bien, pero peor 
               que el modelo FC)

> IceAndFire: modelo FC -> loss=4000, conv 4 capas -> loss=30000, 
              conv 6 capas -> loss=15000

> Catapults: model FC -> loss=3000, conv 4 capas -> loss=40000,
			 conv 6 capas -> loss=55000

En BoulderDash ambos modelos conv son igual de buenos, en IceAndFire tiene mejor training loss
el de 6 capas y en Catapults el de 4 capas.

----

>>> Pruebas a usar Batch Normalization SOLO para los inputs:
> Catapults: El training loss converge más rápido SIN batch normalization (se requieren menos train its.)
			 El training loss al final del entrenamiento (10000 its.) es el mismo en ambos casos.
			 Sin BN, el número de iteraciones para Catapults debería ser de 1000 (o incluso 500).
> IceAndFire: El training loss converge más rápido SIN batch normalization.
              El Q-target y el Q-value es mucho mayor que con batch normalization!!!!!
              <<<<<LOS RESULTADOS EN TEST MEJORAN MUCHÍSIMO!!!!!>>>>>>
              Mejor num de train its. sin BN -> 7500
> BoulderDash: El training loss CON Batch Normalization es mejor, al contrario que los otros
               dos juegos: de 1000 de media de training loss al final (con BN) paso a 3000.
               Los resultados en test empeoran un poco al quitar el batch normalization.

<<No uso Batch Normalization excepto, quizás, para BoulderDash.>>

---- Pruebas de regularización (L2 y dropout) (solo uso BN para los inputs de aquí en adelante):

>>> L2 reg = 0.1
> BoulderDash: ambas gráficas de entrenamiento son casi idénticas. Los resultados en test
               son prácticamente idénticos también.
> IceAndFire: empeora el training loss al usar L2 regularization (se dobla el training loss).
              Las gráficas de Q-target y Q-value también son diferentes. Los resultados en test
              empeoran bastante!!!
> Catapults: las gráficas de entrenamiento son muy parecidas. Empeoran los resultados de test.

<<NO USO REGULARIZACIÓN L2!!!>>

>>> Dropout = 0.1 (solo en las capas convolucionales)
> BoulderDash: el training loss no varía, pero el Q-target y Q-val sí!! Empeoran los resultados
               en test!!
> IceAndFire: el training loss no varía. Empeoran los resultados de test.
> Catapults: el training loss no varía. Resuelve 3 niveles en test en vez de 2 (creo que es suerte).

Dropout no mejora los resultados en test pero hace que las gráficas de entrenamiento tengan más "ruido".
<<NO USO DROPOUT>>

---- Comparo en test a usar 3 vs 4 vs 6 capas conv

>>> 3 capas vs 4 capas:
Las gráficas de entrenamiento de los tres juegos son muy parecidas en ambos casos.



CAMBIAR DE TEST A VALIDACIÓN!!!!

"""

# <Execution mode of the script>
# "validation" -> trains and validates on 5 levels not used for training
# "test" -> trains and tests on the 5 test levels
script_execution_mode = "test"

# <Goal Selection Mode>
# "best" -> use the trained model to select the best subgoal at each state
# "random" -> select subgoals randomly. This corresponds to the Random Model
goal_selection_mode = "best"

# <Seed>
# Used for repetibility
seed=28912 # 28912 (No cambiar la seed!)

# <Model Hyperparameters>
# This script trains and validates one model per each different combination of
# these hyperparameters




# PROBAR QUE SE EJECUTAN LOS NIVELES DE IGNACIO!

# PROBAR A APLICAR PADDING PARA QUE TODOS LOS JUEGOS TENGAN LA MISMA
# DIMENSIÓN!!!

# Probar a disminuir el valor de la penalización cuando se elige un
# subobjetivo no alcanzable o muere (1000)

# QUIZÁS TENGA QUE AÑADIR AL ONE-HOT-MATRIX EL NÚMERO DE GEMAS,
# MONEDAS Y BOTAS QUE HA COGIDO EL AGENTE!!!


""" <<<IMPORTANTE>>>
CUANDO IGNACIO ME DE LOS NUEVOS NIVELES Y TENGA QUE VOLVER A CREAR EL DATASET,
AÑADIR AL ONE-HOT MATRIX LA INFORMACIÓN ADICIONAL SOBRE LOS RECURSOS
OBTENIDOS POR EL AGENTE!
"""

# CAMBIAR LEARNING MODEL EN AGENT.PY Y SAME A VALID!!!

# Architecture
# First conv layer
l1_num_filt = [32] # 32
l1_filter_structure = [ [[3,3],[1,1],"SAME"] ] # [[4,4],[1,1],"VALID"] 

# Second conv layer
l2_num_filt = [32] # 32
l2_filter_structure = [ [[3,3],[1,1],"SAME"] ] # [[4,4],[1,1],"VALID"] 

# Third conv layer
l3_num_filt = [64] # 32 
l3_filter_structure = [ [[3,3],[1,1],"SAME"] ] # [[4,4],[1,1],"VALID"] 

# Fourth conv layer
l4_num_filt = [64] # 32
l4_filter_structure = [ [[3,3],[1,1],"SAME"] ] # [[4,4],[1,1],"VALID"] 

# Fifth conv layer
l5_num_filt = [-1] # 64
l5_filter_structure = [ [[3,3],[1,1],"VALID"] ]

# Sixth conv layer
l6_num_filt = [-1] # 64
l6_filter_structure = [ [[3,3],[1,1],"VALID"] ]

# A single fc layer works better!
fc_num_unis = [[32, 1]] # [32,1] # Number of units of the first and second fully-connected layers

# Training params
tau=[10] # Update period of the target network
alfa = [0.0001] # Learning rate # 0.0001
dropout = [0.0] # Dropout value
batch_size = [32] # 32
use_BN = [False] # If True, Batch Normalization is applied after each conv layer for all the games.
                 # If False, BN is only applied to BoulderDash (BoulderDash ALWAYS uses BN)
# Extra params
# games_to_play = ['BoulderDash', 'IceAndFire', 'Catapults']
games_to_play = ['BoulderDash', 'IceAndFire', 'Catapults']

# For each size, a different model is trained and tested on this number of levels
datasets_sizes_for_training_BoulderDash = [25] # 25 # 20
datasets_sizes_for_training_IceAndFire = [50] # 50 # 45
datasets_sizes_for_training_Catapults = [100] # 100 # 95

# Number of iterations for training
num_its_BoulderDash = [5000] # 5000 # 10000
num_its_IceAndFire = [7500] # 7500 # 2500
num_its_Catapults = [2500] # 2500 # 2500

# Around 19 rep. per night (three games)
ini_rep_model = 0 # Index of the first repetition (0 by default)
repetitions_per_model = 18 # 15 # Each model is trained this number of times

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
models_params_prev = [ [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s]
					for a in l1_num_filt for b in l1_filter_structure for c in l2_num_filt for d in l2_filter_structure \
 					for e in l3_num_filt for f in l3_filter_structure for g in l4_num_filt for h in l4_filter_structure \
 					for i in l5_num_filt for j in l5_filter_structure for k in l6_num_filt for l in l6_filter_structure \
 					for m in fc_num_unis for n in alfa for o in dropout for p in batch_size for q in use_BN \
					for r in tau for s in games_to_play]

# Add the corresponding dataset sizes for each game
models_params_prev_2 = []

for par in models_params_prev:
	if par[-1] == 'BoulderDash':
		for dataset_size in datasets_sizes_for_training_BoulderDash:
			models_params_prev_2.append(par + [dataset_size])

	elif par[-1] == 'IceAndFire':
		for dataset_size in datasets_sizes_for_training_IceAndFire:
			models_params_prev_2.append(par + [dataset_size])

	else: # Catapults
		for dataset_size in datasets_sizes_for_training_Catapults:
			models_params_prev_2.append(par + [dataset_size])

# Add the corresponding training its for each game
models_params = []

for par in models_params_prev_2:
	if par[-2] == 'BoulderDash':
		for num_its in num_its_BoulderDash:
			curr_par = par[:] # Copy by value, not by reference
			curr_par.insert(13, num_its)
			models_params.append(curr_par)

	elif par[-2] == 'IceAndFire':
		for num_its in num_its_IceAndFire:
			curr_par = par[:]
			curr_par.insert(13, num_its)
			models_params.append(curr_par)

	else: # 'Catapults'
		for num_its in num_its_Catapults:
			curr_par = par[:]
			curr_par.insert(13, num_its)
			models_params.append(curr_par)

"""
for i in models_params:
	print(i)

sys.exit()
"""

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
		curr_l5_num_filt = curr_model_params[8]
		curr_l5_filter_structure = curr_model_params[9]
		curr_l6_num_filt = curr_model_params[10]
		curr_l6_filter_structure = curr_model_params[11]
		curr_fc_num_unis = curr_model_params[12]
		curr_num_its = curr_model_params[13]
		curr_alfa = curr_model_params[14]
		curr_dropout = curr_model_params[15]
		curr_batch_size = curr_model_params[16]
		curr_use_BN = curr_model_params[17]
		curr_tau = curr_model_params[18]
		curr_game = curr_model_params[19]
		dataset_size_for_training = curr_model_params[20]

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

		agent_file = re.sub(r'self.l5_num_filt=.*', 'self.l5_num_filt={}'.format(curr_l5_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l5_window=.*', 'self.l5_window={}'.format(curr_l5_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l5_strides=.*', 'self.l5_strides={}'.format(curr_l5_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l5_padding_type=.*', 'self.l5_padding_type="{}"'.format(curr_l5_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l6_num_filt=.*', 'self.l6_num_filt={}'.format(curr_l6_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l6_window=.*', 'self.l6_window={}'.format(curr_l6_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l6_strides=.*', 'self.l6_strides={}'.format(curr_l6_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l6_padding_type=.*', 'self.l6_padding_type="{}"'.format(curr_l6_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.fc_num_unis=.*', 'self.fc_num_unis={}'.format(curr_fc_num_unis), agent_file, count=1)
		agent_file = re.sub(r'self.learning_rate=.*', 'self.learning_rate={}'.format(curr_alfa), agent_file, count=1)
		agent_file = re.sub(r'self.dropout_prob=.*', 'self.dropout_prob={}'.format(curr_dropout), agent_file, count=1)
		agent_file = re.sub(r'self.num_train_its=.*', 'self.num_train_its={}'.format(curr_num_its), agent_file, count=1)
		agent_file = re.sub(r'self.batch_size=.*', 'self.batch_size={}'.format(curr_batch_size), agent_file, count=1)
		agent_file = re.sub(r'self.max_tau=.*', 'self.max_tau={}'.format(curr_tau), agent_file, count=1)
		agent_file = re.sub(r'self.use_BN=.*', 'self.use_BN={}'.format(curr_use_BN), agent_file, count=1)

		# Change other variables
		agent_file = re.sub(r'self.game_playing=.*', 'self.game_playing="{}"'.format(curr_game), agent_file, count=1)
		agent_file = re.sub(r'self.dataset_size_for_training=.*', 'self.dataset_size_for_training={}'.format(dataset_size_for_training), agent_file, count=1)

		# Change goal_selection_mode
		agent_file = re.sub(r'self.goal_selection_mode=.*', 'self.goal_selection_mode="{}"'.format(goal_selection_mode), agent_file, count=1)

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
		for curr_rep in range(ini_rep_model, ini_rep_model+repetitions_per_model):
			# <Calculate the seed for the current execution>
			curr_seed = seed*(curr_rep+1) % 1000000

			# <Create the model name using the hyperparameters values>

			if script_execution_mode == "validation":
				# The name of the model can't be that long!! (it raises an exception on tensorflow)
				curr_model_name = "DQN_Pruebas-conv-loss_BN-solo-inputs_Dropout-0.1_val_c1-{}_c2-{}_c3-{}_c4-{}_c5-{}_c6-{}_fc-{}_{}_its-{}_{}_{}". \
								format(curr_l1_num_filt, curr_l2_num_filt, curr_l3_num_filt, curr_l4_num_filt, curr_l5_num_filt, curr_l6_num_filt, \
									   curr_fc_num_unis[0], curr_fc_num_unis[1], curr_num_its, curr_game, curr_rep)

				"""curr_model_name = "DQN_Full-FC-7-layers_Dropout-0.5_its-{}_{}_{}". \
								format(curr_num_its, curr_game, curr_rep)"""

			else:
				curr_model_name = "DQN_pruebas_max_pooling_y_padding_SAME_its-{}_{}_{}". \
								format(curr_num_its, curr_game, curr_rep)
				# curr_model_name = "DQN_prueba_overfitting_train_y_test-{}".format(curr_game)

			print("\n\nCurrent model: {} - Current repetition: {}\n".format(curr_model_name, curr_rep))

			# <Change Agent.py>	

			# Load file in memory
			with open('MyAgent/Agent.py', 'r') as file:
				agent_file = file.read()

			# Change the model name
			agent_file = re.sub(r'self.network_name=.*', 'self.network_name="{}"'.format(curr_model_name), agent_file, count=1)

			# Change the seed
			agent_file = re.sub(r'self.level_seed=.*', 'self.level_seed={}'.format(curr_seed), agent_file, count=1)

			# Save file
			with open('MyAgent/Agent.py', 'w') as file:
				file.write(agent_file)


			# ------ TRAINING ------

			# Skip training if we are testing the random model
			if goal_selection_mode == "best":
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

			# Select five validation levels using the random seed
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

				# Set the seed for repetibility
				random.seed(curr_seed)

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


					