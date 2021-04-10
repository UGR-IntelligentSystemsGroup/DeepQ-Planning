# Given the output of the test_output_parser.py script, it obtains the average number of actions
# (for BoulderDash and IceAndFire) or success rate (for Catapults) of a model as the number of 
# training iterations increases

import re
import numpy as np

input_file = "test_output_parsed.txt"

# BoulderDash, IceAndFire or Catapults
game_playing = "Catapults"

num_train_levels = 5 # Train levels are indexes 0 to num_train_levels-1. The rest are test levels.

with open(input_file, 'r') as f:
	input_data = f.read()

# Obtain the different models that exist (without taking into account the
# number of iterations)

# Group 1 -> number of the network
# Group 2 -> game played
match_list = re.findall(r'(DQN.*)_its-[0-9]+_(.*) - ', input_data)

# Convert it to a set to get the unique values (model name and game played)
match_set = set(match_list)

# For each different model, obtain the average number of actions as the train its increase
for curr_match in match_set:
	# List of the different number of iterations
	# Group 1 (of this new match) -> number of iterations
	curr_model_match_list_its = re.findall(r'{}_its-[0-9]+_{} - .*lvs=[0-9]+-([0-9]+)'.format(curr_match[0], curr_match[1]),
	 input_data)

	# List of lists of the number of actions / success rate
	# Group 1 (of this new match) -> list with number of actions / success rate
	if game_playing == "Catapults": # Catapults -> use success rate
		curr_model_match_list_num_actions = re.findall(r'Success rate per level: \[(.+)\]', input_data)
	else: # BoulderDash / IceAndFire -> use number of actions
		curr_model_match_list_num_actions = re.findall(r'Average num actions per level: \[(.+)\]', input_data)

	its_list = [int(i) for i in curr_model_match_list_its]
	# Calculate the actions/success rate for each level for each element in curr_model_match_list_num_actions
	actions_list = [np.array(action_list.replace(" ", "").split(',')).astype('float') for action_list
	 in curr_model_match_list_num_actions]

	# Matrix where rows corresponds to different number of its and columns to different levels
	actions_matrix = np.array(actions_list, dtype="float").reshape((-1,len(actions_list[0])))

	# Only for number of actions
	if game_playing != "Catapults":
		# Normalize each column (level) to mean=0,std=1 by substracting the mean and dividing by standard deviation
		for j in range(actions_matrix.shape[1]):
			actions_matrix[:,j] = (actions_matrix[:,j] - np.mean(actions_matrix[:,j])) / np.std(actions_matrix[:,j])

	# Calculate the normalized average number of actions / success rate for each different number of its
	# norm_averaged_num_actions = np.around(np.mean(actions_matrix, axis=1), decimals=2) # Only use 2 decimals
	norm_averaged_num_actions_train = np.around(np.mean(actions_matrix[:,0:num_train_levels], axis=1), decimals=2)
	norm_averaged_num_actions_test = np.around(np.mean(actions_matrix[:,num_train_levels:], axis=1), decimals=2)

	# Order by number of iterations
	results_list_train = sorted(zip(its_list, norm_averaged_num_actions_train))
	results_list_test = sorted(zip(its_list, norm_averaged_num_actions_test))

	# Print the number of iterations and average number of actions per each num its
	results_list_its_train = [i[0] for i in results_list_train]
	results_list_num_act_train = [i[1] for i in results_list_train]
	results_list_num_act_test = [i[1] for i in results_list_test]

	print("\n----- {} - {} -----".format(curr_match[0], curr_match[1]))
	print("> Num its: ", results_list_its_train)
	if game_playing == "Catapults":
		print("\n> Train Success Rate: ", results_list_num_act_train)
		print("\n> Test Success Rate: ", results_list_num_act_test)
	else:
		print("> Num actions: ", results_list_num_act)