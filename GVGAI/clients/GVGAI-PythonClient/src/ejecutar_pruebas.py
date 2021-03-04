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


"""

# <Execution mode of the script>
# "validation" -> trains and validates on 5 levels not used for training
# "test" -> trains and tests on all the test levels
script_execution_mode = "test"

# <Goal Selection Mode>
# "best" -> use the trained model to select the best subgoal at each state
# "random" -> select subgoals randomly. This corresponds to the Random Model
# "greedy" -> plan for each subgoal and select the valid subgoal with the shortest plan
goal_selection_mode = "best"

# <Seed>
# Used for repetibility
seed=28912 # 28912 (No cambiar la seed!)

# <Model Hyperparameters>
# This script trains and validates one model per each different combination of
# these hyperparameters


"""
Pruebas
> Entre [512,128,32,1] y [1024,256,32,1] ambos obtienen casi los mismos resultados en test!!!

> Entre 6 capas y 8 capas conv (con 256 filtros las 2 últimas), se obtienen mejores resultados en 
test con 8 capas, aunque las gráficas de entrenamiento tienen más picos.

> Entre 15000 y 20000 training its. Se obtienen resultados un poco mejores (en test) con 
	20000 training its.

> Al probar dropout, los resultados al usar dropout 0.2 y no usarlo son casi idénticos. Al usar
dropout 0.4 empeoran los resultados -> no uso dropout

> IceAndFire y Catapults funcionan mejor SIN Batch Normalization!!
	(las gráficas de Q-value y Q-target son diferentes con BN y en IceAndFire
	 los resultados en test son peores)
	 BoulderDash funciona mejor CON BN!!!

> Pruebas mejor num fc tres juegos.
	Al aumentar el número de unidades fc en BoulderDash, el loss no disminuye!!
	Las gráficas de Q-value y Q-target sí varían un poco.
	Lo mismo pasa en IceAndFire (de hecho al aumentar las unidades fc más allá
	de [128, 32] el entrenamiento es más inestable al igual que las gráficas
	de Q_val y Q-target).
	Lo mismo ocurre con Catapults.

	Respecto a test, para IceAndFire y Catapults los mejores resultados son con el
	menor número de unidades fc (128, 32). En BoulderDash, funciona mejor usar
	[512, 128, 32] unidades fc, obteniendo un 7% menos de acciones de media que con
	128, 32. 

> Pruebas 6 capas 128 vs 8 capas 256.
	Con 8 capas el training loss es mejor y en test, para boulderdash da igual, para iceandfire
	es un poco mejor y en Catapults es también un poco mejor -> uso 8 capas.

> Al probar a aumentar las capas a 10 (usando 256 filtros en las últimas), los resultados
	empeoran en BoulderDash y Catapults, y se mantienen igual en IceAndFire.

> Al entrenar Catapults sobre solo los 100 niveles antiguos, los resultados empeoran.

> Pruebas mejor num its.
	BoulderDash: no está claro cuál es mejor -> Me quedo con 20000
	IceAndFire: no está claro cuál es mejor -> Me quedo con 20000
	Catapults: 25000 repeticiones.

> El mejor número de filtros en las últimas capas conv es 128, no 64 o 256.

>> Pruebas tau=500 <<

alfa-0.00025 es demasiado alto!

EN BOULDERDASH ES A PARTIR DE 100K TRAINING ITS CUANDO EL Q-TARGET Y Q-VALUE
EMPIEZA A DISMINUIR POR DEBAJO DE 0!!!

¡Parece que sí se puede producir overfitting por usar demasiadas training its!
Los resultados en BoulderDash con 40000 iteraciones son mejores que con 200000.
En IceAndFire los resultados con 200000 iteraciones son mejores que con 40000.
En IceAndFire los resultados con 100000 y 200000 iteraciones son casi idénticos.
<<ENTRENO EN BOULDERDASH CON 40000 ITERACIONES Y EN ICEANDFIRE CON 100000.>>
Entreno también con 100000 its en Catapults (aunque sería necesario hacer más pruebas).

>> Pruebas PER <<
Las losses abs y huber no funcionan bien (en IceAndFire el Q-value no disminuye
por debajo de 0 sino que sigue aumentando)

Con alfa=0.0001 (el que estábamos usando) en IceAndFire parece que el mejor número
de its es 400000 (en ese número ya hay pocos picos y el loss está por debajo de 10)

Con alfa=0.000025 también converge el loss pero mucho más lento y pero (<<el mejor
alfa es 0.0001>>)

En BoulderDash, para que converja es necesario usar alfa=0.00005. Parece que el mejor
num de its es 300000.

>> Pruebas modelos simples <<

>> Modelo con 512 unidades fc, 3 capas conv. (modelo simple)
Al probar con el modelo de DQN Nature (3 capas conv, 512 unidades fc), es capaz de converger en IceAndFire
en 1M train its. El error converge muy rápido. Sin embargo, el Q-target y Q-value tardan en descender
lo suficiente (en el modelo complejo, ambas gráficas disminuyen por debajo de 0 mucho más rápido).

Los resultados en test (tras 1M train its) son prácticamente idénticos (un 2% de más acciones en todos los
niveles de media, aunque tiene menos errores) que el modelo complejo (con 1M de train its y PER también).

Tras el mejor número de its (1.25M), el modelo simple es mejor que el complejo (este último con su número
óptimo de iteraciones también) -> usa un 96% del número de acciones y comete menos errores
"""


# <Architecture>

# Complex architecture
# 8 conv layers -> it is able to converge on all three games
# fc units -> [128,32,1,1]
"""
l1_num_filt = [32]
l1_filter_structure = [ [[3,3],[1,1],"SAME"] ]
l2_num_filt = [32]
l2_filter_structure = [ [[3,3],[1,1],"SAME"] ] 

l3_num_filt = [64]
l3_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l4_num_filt = [64]
l4_filter_structure = [ [[3,3],[1,1],"VALID"] ] 
l5_num_filt = [64]
l5_filter_structure = [ [[3,3],[1,1],"VALID"] ]

l6_num_filt = [128]
l6_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l7_num_filt = [128]
l7_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l8_num_filt = [128]
l8_filter_structure = [ [[3,3],[1,1],"VALID"] ]
"""

# Simple architecture (inspired by DQN architecture of original paper)
# 3 conv layers
# fc units -> [[512,1,1,1]] 
l1_num_filt = [32]
l1_filter_structure = [ [[5,5],[1,1],"VALID"] ]
l2_num_filt = [64]
l2_filter_structure = [ [[5,5],[1,1],"VALID"] ] 
l3_num_filt = [64]
l3_filter_structure = [ [[5,5],[1,1],"VALID"] ]

l4_num_filt = [-1]
l4_filter_structure = [ [[3,3],[1,1],"VALID"] ] 
l5_num_filt = [-1]
l5_filter_structure = [ [[3,3],[1,1],"VALID"] ]

l6_num_filt = [-1]
l6_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l7_num_filt = [-1]
l7_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l8_num_filt = [-1]
l8_filter_structure = [ [[3,3],[1,1],"VALID"] ]


# Architecture with 5 conv layers
# fc units -> [[128,1,1,1]] 
"""
l1_num_filt = [32]
l1_filter_structure = [ [[5,5],[1,1],"VALID"] ]
l2_num_filt = [32]
l2_filter_structure = [ [[3,3],[1,1],"VALID"] ] 
l3_num_filt = [64]
l3_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l4_num_filt = [64]
l4_filter_structure = [ [[3,3],[1,1],"VALID"] ] 
l5_num_filt = [64]
l5_filter_structure = [ [[3,3],[1,1],"VALID"] ]

l6_num_filt = [-1]
l6_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l7_num_filt = [-1]
l7_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l8_num_filt = [-1]
l8_filter_structure = [ [[3,3],[1,1],"VALID"] ]
"""

l9_num_filt = [-1]
l9_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l10_num_filt = [-1]
l10_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l11_num_filt = [-1]
l11_filter_structure = [ [[3,3],[1,1],"SAME"] ]
l12_num_filt = [-1]
l12_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l13_num_filt = [-1]
l13_filter_structure = [ [[3,3],[1,1],"SAME"] ]
l14_num_filt = [-1]
l14_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l15_num_filt = [-1]
l15_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l16_num_filt = [-1]
l16_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l17_num_filt = [-1]
l17_filter_structure = [ [[3,3],[1,1],"SAME"] ]
l18_num_filt = [-1]
l18_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l19_num_filt = [-1]
l19_filter_structure = [ [[3,3],[1,1],"VALID"] ]
l20_num_filt = [-1]
l20_filter_structure = [ [[3,3],[1,1],"VALID"] ]

# Number of units of the fully-connected layers
fc_num_unis = [[128,1,1,1]] 

# Training params
tau=[1000] # 10 # Update period of the target network
# CHANGE FOR BOULDERDASH!
alfa = [0.00005] # 0.0001 # 0.00005 for BoulderDash # Learning rate
gamma = [1] # 1 # Discount rate for rewards
dropout = [0.0] # Dropout value
batch_size = [32] # 32
use_BN = [False] # If True, Batch Normalization is applied after each conv layer for all the games.
								 # If False, BN is only applied to BoulderDash (BoulderDash ALWAYS uses BN)
# Extra params
# games_to_play = ['BoulderDash', 'IceAndFire', 'Catapults']
games_to_play = ['BoulderDash']

# For each size, a different model is trained and tested on this number of levels
datasets_sizes_for_training_BoulderDash = [100]
datasets_sizes_for_training_IceAndFire = [100]
datasets_sizes_for_training_Catapults = [200]

# Number of iterations for training
num_its_BoulderDash = [500000] # After 500000 its, results are always bad # 1000000 # 40000 # 20000 
num_its_IceAndFire = [1500000] # 400000 # 100000 # 20000 # Creo que el mejor número de its es 400000
num_its_Catapults = [1500000] # 100000 # 20000 # Creo que el mejor número de its es 300000
# 1 hour -> 1 rep. for every game
ini_rep_model = 1 # Index of the first repetition
repetitions_per_model = 1 # 15 # Each model is trained this number of times

# Test level indexes
# If script_execution_mode == "test" these are the indexes of the levels to use
# for testing the trained model (or random model)
# The need to be grouped in pairs (or as a one-element tuple)
# test_level_indexes = [(5,6),(7,8),(9,10)]
# test_level_indexes = [(0,1),(2,3),(4,)] # Use this one for validation
# test_level_indexes = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,)]
test_level_indexes = [(0,1),(2,3),(4,5),(6,7),(8,9),(10,)]

# If False, each saved model is only tested at the end of the training
# If True, each saved model is tested every "test_it_interval" training its
test_all_its = True
test_it_interval = 50000 # 20000

# If True, the train phase is skipped (we assume the model has already been trained and saved)
skip_train = False

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
models_params_prev = [ [a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z,aa,bb,cc,dd,ee,ff,gg,hh,ii,jj,kk,ll,mm,nn,oo,pp,qq,rr,ss,tt,uu,vv]
					for a in l1_num_filt for b in l1_filter_structure for c in l2_num_filt for d in l2_filter_structure \
					for e in l3_num_filt for f in l3_filter_structure for g in l4_num_filt for h in l4_filter_structure \
					for i in l5_num_filt for j in l5_filter_structure for k in l6_num_filt for l in l6_filter_structure \
					for m in l7_num_filt for n in l7_filter_structure for o in l8_num_filt for p in l8_filter_structure \
					for q in l9_num_filt for r in l9_filter_structure for s in l10_num_filt for t in l10_filter_structure \
					for u in l11_num_filt for v in l11_filter_structure for w in l12_num_filt for x in l12_filter_structure \
					for y in l13_num_filt for z in l13_filter_structure for aa in l14_num_filt for bb in l14_filter_structure \
					for cc in l15_num_filt for dd in l15_filter_structure for ee in l16_num_filt for ff in l16_filter_structure \
					for gg in l17_num_filt for hh in l17_filter_structure for ii in l18_num_filt for jj in l18_filter_structure \
					for kk in l19_num_filt for ll in l19_filter_structure for mm in l20_num_filt for nn in l20_filter_structure \
					for oo in fc_num_unis for pp in alfa for qq in gamma for rr in dropout for ss in batch_size for tt in use_BN \
					for uu in tau for vv in games_to_play]

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
			curr_par.insert(41, num_its) # Aumentar en 2*el número de capas añadidas
			models_params.append(curr_par)

	elif par[-2] == 'IceAndFire':
		for num_its in num_its_IceAndFire:
			curr_par = par[:]
			curr_par.insert(41, num_its)
			models_params.append(curr_par)

	else: # 'Catapults'
		for num_its in num_its_Catapults:
			curr_par = par[:]
			curr_par.insert(41, num_its)
			models_params.append(curr_par)

"""
for i in models_params:
	print(i)

sys.exit()"""


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
		curr_l7_num_filt = curr_model_params[12]
		curr_l7_filter_structure = curr_model_params[13]
		curr_l8_num_filt = curr_model_params[14]
		curr_l8_filter_structure = curr_model_params[15]
		curr_l9_num_filt = curr_model_params[16]
		curr_l9_filter_structure = curr_model_params[17]
		curr_l10_num_filt = curr_model_params[18]
		curr_l10_filter_structure = curr_model_params[19]
		curr_l11_num_filt = curr_model_params[20]
		curr_l11_filter_structure = curr_model_params[21]
		curr_l12_num_filt = curr_model_params[22]
		curr_l12_filter_structure = curr_model_params[23]
		curr_l13_num_filt = curr_model_params[24]
		curr_l13_filter_structure = curr_model_params[25]
		curr_l14_num_filt = curr_model_params[26]
		curr_l14_filter_structure = curr_model_params[27]
		curr_l15_num_filt = curr_model_params[28]
		curr_l15_filter_structure = curr_model_params[29]
		curr_l16_num_filt = curr_model_params[30]
		curr_l16_filter_structure = curr_model_params[31]
		curr_l17_num_filt = curr_model_params[32]
		curr_l17_filter_structure = curr_model_params[33]
		curr_l18_num_filt = curr_model_params[34]
		curr_l18_filter_structure = curr_model_params[35]
		curr_l19_num_filt = curr_model_params[36]
		curr_l19_filter_structure = curr_model_params[37]
		curr_l20_num_filt = curr_model_params[38]
		curr_l20_filter_structure = curr_model_params[39]
		curr_fc_num_unis = curr_model_params[40]
		curr_num_its = curr_model_params[41]
		curr_alfa = curr_model_params[42]
		curr_gamma = curr_model_params[43]
		curr_dropout = curr_model_params[44]
		curr_batch_size = curr_model_params[45]
		curr_use_BN = curr_model_params[46]
		curr_tau = curr_model_params[47]
		curr_game = curr_model_params[48]
		dataset_size_for_training = curr_model_params[49]

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

		agent_file = re.sub(r'self.l7_num_filt=.*', 'self.l7_num_filt={}'.format(curr_l7_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l7_window=.*', 'self.l7_window={}'.format(curr_l7_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l7_strides=.*', 'self.l7_strides={}'.format(curr_l7_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l7_padding_type=.*', 'self.l7_padding_type="{}"'.format(curr_l7_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l8_num_filt=.*', 'self.l8_num_filt={}'.format(curr_l8_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l8_window=.*', 'self.l8_window={}'.format(curr_l8_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l8_strides=.*', 'self.l8_strides={}'.format(curr_l8_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l8_padding_type=.*', 'self.l8_padding_type="{}"'.format(curr_l8_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l9_num_filt=.*', 'self.l9_num_filt={}'.format(curr_l9_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l9_window=.*', 'self.l9_window={}'.format(curr_l9_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l9_strides=.*', 'self.l9_strides={}'.format(curr_l9_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l9_padding_type=.*', 'self.l9_padding_type="{}"'.format(curr_l9_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l10_num_filt=.*', 'self.l10_num_filt={}'.format(curr_l10_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l10_window=.*', 'self.l10_window={}'.format(curr_l10_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l10_strides=.*', 'self.l10_strides={}'.format(curr_l10_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l10_padding_type=.*', 'self.l10_padding_type="{}"'.format(curr_l10_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l11_num_filt=.*', 'self.l11_num_filt={}'.format(curr_l11_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l11_window=.*', 'self.l11_window={}'.format(curr_l11_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l11_strides=.*', 'self.l11_strides={}'.format(curr_l11_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l11_padding_type=.*', 'self.l11_padding_type="{}"'.format(curr_l11_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l12_num_filt=.*', 'self.l12_num_filt={}'.format(curr_l12_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l12_window=.*', 'self.l12_window={}'.format(curr_l12_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l12_strides=.*', 'self.l12_strides={}'.format(curr_l12_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l12_padding_type=.*', 'self.l12_padding_type="{}"'.format(curr_l12_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l13_num_filt=.*', 'self.l13_num_filt={}'.format(curr_l13_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l13_window=.*', 'self.l13_window={}'.format(curr_l13_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l13_strides=.*', 'self.l13_strides={}'.format(curr_l13_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l13_padding_type=.*', 'self.l13_padding_type="{}"'.format(curr_l13_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l14_num_filt=.*', 'self.l14_num_filt={}'.format(curr_l14_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l14_window=.*', 'self.l14_window={}'.format(curr_l14_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l14_strides=.*', 'self.l14_strides={}'.format(curr_l14_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l14_padding_type=.*', 'self.l14_padding_type="{}"'.format(curr_l14_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l15_num_filt=.*', 'self.l15_num_filt={}'.format(curr_l15_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l15_window=.*', 'self.l15_window={}'.format(curr_l15_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l15_strides=.*', 'self.l15_strides={}'.format(curr_l15_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l15_padding_type=.*', 'self.l15_padding_type="{}"'.format(curr_l15_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l16_num_filt=.*', 'self.l16_num_filt={}'.format(curr_l16_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l16_window=.*', 'self.l16_window={}'.format(curr_l16_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l16_strides=.*', 'self.l16_strides={}'.format(curr_l16_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l16_padding_type=.*', 'self.l16_padding_type="{}"'.format(curr_l16_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l17_num_filt=.*', 'self.l17_num_filt={}'.format(curr_l17_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l17_window=.*', 'self.l17_window={}'.format(curr_l17_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l17_strides=.*', 'self.l17_strides={}'.format(curr_l17_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l17_padding_type=.*', 'self.l17_padding_type="{}"'.format(curr_l17_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l18_num_filt=.*', 'self.l18_num_filt={}'.format(curr_l18_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l18_window=.*', 'self.l18_window={}'.format(curr_l18_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l18_strides=.*', 'self.l18_strides={}'.format(curr_l18_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l18_padding_type=.*', 'self.l18_padding_type="{}"'.format(curr_l18_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l19_num_filt=.*', 'self.l19_num_filt={}'.format(curr_l19_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l19_window=.*', 'self.l19_window={}'.format(curr_l19_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l19_strides=.*', 'self.l19_strides={}'.format(curr_l19_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l19_padding_type=.*', 'self.l19_padding_type="{}"'.format(curr_l19_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.l20_num_filt=.*', 'self.l20_num_filt={}'.format(curr_l20_num_filt), agent_file, count=1)
		agent_file = re.sub(r'self.l20_window=.*', 'self.l20_window={}'.format(curr_l20_filter_structure[0]), agent_file, count=1)
		agent_file = re.sub(r'self.l20_strides=.*', 'self.l20_strides={}'.format(curr_l20_filter_structure[1]), agent_file, count=1)
		agent_file = re.sub(r'self.l20_padding_type=.*', 'self.l20_padding_type="{}"'.format(curr_l20_filter_structure[2]), agent_file, count=1)

		agent_file = re.sub(r'self.fc_num_unis=.*', 'self.fc_num_unis={}'.format(curr_fc_num_unis), agent_file, count=1)
		agent_file = re.sub(r'self.learning_rate=.*', 'self.learning_rate={}'.format(curr_alfa), agent_file, count=1)
		agent_file = re.sub(r'self.gamma=.*', 'self.gamma={}'.format(curr_gamma), agent_file, count=1)
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

			# *** Name ***
			if script_execution_mode == "validation":
				# The name of the model can't be that long!! (it raises an exception on tensorflow)

				curr_model_name = "DQN_Final_Model_val_fc-{}_{}_{}_{}_its-{}_{}_{}". \
								format(curr_fc_num_unis[0], curr_fc_num_unis[1], curr_fc_num_unis[2],
								 curr_fc_num_unis[3], curr_num_its, curr_game, curr_rep)

			else:
				curr_model_name = "DQN_Simple_model_test_gamma-{}_fc-{}_{}_{}_{}_its-{}_{}_{}". \
								format(curr_gamma, curr_fc_num_unis[0], curr_fc_num_unis[1], curr_fc_num_unis[2],
								 curr_fc_num_unis[3], curr_num_its, curr_game, curr_rep)

				# curr_model_name = "DQP_times_test-{}".format(curr_game)

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

			# Skip training if we are testing the random/greedy model or if skip_train is True
			if goal_selection_mode == "best" and skip_train == False:
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

			# Check if we are going to test only the saved model with the largest number of training its
			# of if we are going to test the saved model every "test_it_interval" train its
			if test_all_its == False:
				array_its_to_test = [curr_num_its]
			else:
				array_its_to_test = [i for i in range(test_it_interval, curr_num_its+1, test_it_interval)]

			# Obtain the test results of the model for each number of its
			for array_its_to_test_curr_elem in array_its_to_test:

				# <Change Agent.py>

				# Load file in memory
				with open('MyAgent/Agent.py', 'r') as file:
					agent_file = file.read()

				# Change execution mode
				agent_file = re.sub(r'self.EXECUTION_MODE=.*', 'self.EXECUTION_MODE="test"', agent_file, count=1)
				# Change num of train its (of the model to load)
				agent_file = re.sub(r'self.num_train_its_model=.*', 'self.num_train_its_model={}'.format(array_its_to_test_curr_elem), agent_file, count=1)

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

					# Get the path of the test levels without the index and the ".txt"
					val_levels_path = re.search(r"(\D+)\d+.txt", val_levels[0]).group(1) # \D matches any character which is NOT a digit

				# <Validate the model on a different pair of val/test levels each time>
				for curr_val_levels in test_level_indexes:

					# <Remove the test levels (3-4) of the corresponding game>
					test_levels_current_game = [test_lvs_directory + level_name for level_name in curr_test_lvs]
					
					for level in test_levels_current_game:
						subprocess.call("rm {} 2> /dev/null".format(level), shell=True)

					if len(curr_val_levels) == 1: # Only one validation level to test

						if script_execution_mode == "test":					
							val_level_name = val_levels_path + str(curr_val_levels[0]) + ".txt"
						else:
							val_level_name = val_levels[curr_val_levels[0]]

						print("\nNIVEL:", val_level_name)

						# <Copy the new validation level as the levels 3-4>
						subprocess.call("cp {} {}".format(val_level_name, test_levels_current_game[0]), shell=True) 
						subprocess.call("cp {} {}".format(val_level_name, test_levels_current_game[1]), shell=True) 

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

						if script_execution_mode == "test":	
							val_level1_name = val_levels_path + str(curr_val_levels[0]) + ".txt"
							val_level2_name = val_levels_path + str(curr_val_levels[1]) + ".txt"
						else:
							val_level1_name = val_levels[curr_val_levels[0]]
							val_level2_name = val_levels[curr_val_levels[1]]

						print("\nNIVELES:", val_level1_name, val_level2_name)

						# <Copy the new validation levels as the levels 3-4>
						subprocess.call("cp {} {}".format(val_level1_name, test_levels_current_game[0]), shell=True) 
						subprocess.call("cp {} {}".format(val_level2_name, test_levels_current_game[1]), shell=True) 

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


					