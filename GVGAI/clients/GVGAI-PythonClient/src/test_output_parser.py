"""
This script parses the execution results stored in test_output.txt
"""

import re
import numpy as np

output_file = "test_output.txt"
num_test_levels = 5

def get_average_results_per_game(game_list):
	"""
	This function reads the output file and calculates for each game in
	@game_list the average results: number of errors and actions for
	BoulderDash and IceAndFire and also the success rate for Catapults.
	"""

	with open(output_file, 'r') as f:
		results_str = f.read()

	for curr_game in game_list:
		# Find all the different game models for the current game
		# Each parenthesis represents a different group
		# Group 1 -> Name of the network
		# Group 2 -> Number of levels it was trained on
		match_list = re.findall(r'(DQN.*)_[0-9]+-([0-9]+) \| {} \|.*'.format(curr_game), results_str)

		# Convert it to a set to get the unique values (model names)
		match_set = set(match_list)

		# For each different model of the current game, match the corresponding test/validation results
		for curr_match in match_set:
			# Get all the lines containing the results
			curr_model_match_list = re.findall(r'{}_[0-9]+-{} \| {} \| (-?[0-9]+) \| ([0-9]+) \| (\d+.?\d*) \| (\d+.?\d*)'.
				format(curr_match[0], curr_match[1], curr_game), results_str)

			# Separe the number of errors, actions, total time and mean time
			curr_model_num_errors = [int(match[0]) for match in curr_model_match_list]
			curr_model_num_actions = [int(match[1]) for match in curr_model_match_list]
			curr_model_total_time = [float(match[2]) for match in curr_model_match_list]
			curr_model_mean_time = [float(match[3]) for match in curr_model_match_list]

			if curr_game in ('BoulderDash', 'IceAndFire'):
				# Calculate the average number of errors and actions for each of the
				# five test/validation levels
				average_num_errors_per_level = [np.average(curr_model_num_errors[i::num_test_levels]) for i in range(num_test_levels)]
				average_num_actions_per_level = [np.average(curr_model_num_actions[i::num_test_levels]) for i in range(num_test_levels)]

				# Calculate the average total time and mean time for each of the five
				# test/validation levels
				average_total_time_per_level = [np.average(curr_model_total_time[i::num_test_levels]) for i in range(num_test_levels)]
				average_mean_time_per_level = [np.average(curr_model_mean_time[i::num_test_levels]) for i in range(num_test_levels)]

				print("\n--- {} ---\n".format(curr_game))
				print("{} - lvs={}".format(curr_match[0], curr_match[1])) # Model Name and number of levels
				print("Average num errors per level: ", average_num_errors_per_level)
				print("Average num actions per level: ", average_num_actions_per_level)
				print("Average total time per level: ", average_total_time_per_level)
				print("Average mean time per level: ", average_mean_time_per_level)

			else: # Catapults
				# Calculate success_rate (percentage of levels the agent beats)



				success_rate_per_level = [(len(curr_model_num_errors[i::num_test_levels]) - list(curr_model_num_errors[i::num_test_levels]).count(-1))
				 / len(curr_model_num_errors[i::num_test_levels]) * 100 for i in range(num_test_levels)]

				
				# Calculate the average number of errors per test level <completed>
				average_num_errors_per_completed_level = []
				for i in range(num_test_levels):
					# Obtain the number of errors of the current level
					curr_num_errors = curr_model_num_errors[i::num_test_levels]

					# Delete the -1 entries (only use the errors of the completed levels)
					curr_num_errors = [e for e in curr_num_errors if e != -1]

					# If no level was completed, the average number of errors is -1 (undetermined)
					if len(curr_num_errors) == 0:
						average_num_errors_per_completed_level.append(-1)
					else: # otherwise, we calculate the average
						average_num_errors_per_completed_level.append(np.average(curr_num_errors))

				# Calculate the average number of actions, average total time and average mean time
				# per test level <completed>
				average_num_actions_per_completed_level = []
				average_total_time_per_completed_level = []
				average_mean_time_per_completed_level = []

				for i in range(num_test_levels):
					# Obtain the number of errors, total time and mean time of the current level
					curr_num_actions = curr_model_num_actions[i::num_test_levels]
					curr_total_time = curr_model_total_time[i::num_test_levels]
					curr_mean_time = curr_model_mean_time[i::num_test_levels]

					# Delete the entries of levels with -1 errors (levels not completed)
					curr_num_errors = curr_model_num_errors[i::num_test_levels]
					curr_num_actions = [a for a,e in zip(curr_num_actions, curr_num_errors) if e != -1]
					curr_total_time = [a for a,e in zip(curr_total_time, curr_num_errors) if e != -1]
					curr_mean_time = [a for a,e in zip(curr_mean_time, curr_num_errors) if e != -1]

					# If no level was completed, the average number of actions, total time and
					# mean time is -1 (undetermined)
					if len(curr_num_actions) == 0:
						average_num_actions_per_completed_level.append(-1)
						average_total_time_per_completed_level.append(-1)
						average_mean_time_per_completed_level.append(-1)
					else: # otherwise, we calculate the average
						average_num_actions_per_completed_level.append(np.average(curr_num_actions))
						average_total_time_per_completed_level.append(np.average(curr_total_time))
						average_mean_time_per_completed_level.append(np.average(curr_mean_time))


				print("\n--- {} ---\n".format(curr_game))
				print("{} - lvs={}".format(curr_match[0], curr_match[1])) # Model Name and number of levels
				print("Success rate per level", success_rate_per_level)
				print("Average num errors per completed level: ", average_num_errors_per_completed_level)
				print("Average num actions per completed level: ", average_num_actions_per_completed_level)
				print("Average total time per completed level: ", average_total_time_per_completed_level)
				print("Average mean time per completed level: ", average_mean_time_per_completed_level)

if __name__ == '__main__':
	# games = ['BoulderDash', 'IceAndFire', 'Catapults']
	games = ['BoulderDash', 'IceAndFire', 'Catapults']

	get_average_results_per_game(games)