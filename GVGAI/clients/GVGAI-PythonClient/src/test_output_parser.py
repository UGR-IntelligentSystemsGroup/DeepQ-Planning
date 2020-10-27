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
		# Match all the lines corresponding to the current game
		match_str = re.findall(r'.+\| {} \|.+'.format(curr_game), results_str)

		# Separate match_str into the number of errors and number of actions
		num_errors_array = np.array([int(line_str.replace(' ', '').split('|')[2]) for line_str in match_str])
		num_actions_array = np.array([int(line_str.replace(' ', '').split('|')[3]) for line_str in match_str])

		if curr_game in ('BoulderDash', 'IceAndFire'):
			# Calculate the average number of errors and actions for each of the
			# five test/validation levels
			average_num_errors_per_level = [np.average(num_errors_array[i::num_test_levels]) for i in range(num_test_levels)]
			average_num_actions_per_level = [np.average(num_actions_array[i::num_test_levels]) for i in range(num_test_levels)]

			print("\n--- {} ---\n".format(curr_game))
			print("Average num errors per level: ", average_num_errors_per_level)
			print("Average num actions per level: ", average_num_actions_per_level)

		else: # Catapults
			# Calculate success_rate (percentage of levels the agent beats)
			success_rate_per_level = [(len(num_errors_array[i::num_test_levels]) - list(num_errors_array[i::num_test_levels]).count(-1))
			 / len(num_errors_array[i::num_test_levels]) * 100 for i in range(num_test_levels)]

			# Calculate the average number of errors per test level <completed>
			average_num_errors_per_completed_level = [np.average(num_errors_array[i::num_test_levels][num_errors_array[i::num_test_levels]!=-1])
			 for i in range(num_test_levels)] # Calculate the average num errors of the different executions of the game level where the number of errors is different from -1

			# Calculate the average number of actions per test level <completed>
			average_num_actions_per_completed_level = [np.average(num_actions_array[i::num_test_levels][num_errors_array[i::num_test_levels]!=-1])
			 for i in range(num_test_levels)] # Calculate the average num actions of the different executions of the game level where the number of errors is different from -1

			print("\n--- {} ---\n".format(curr_game))
			print("Success rate per level", success_rate_per_level)
			print("Average num errors per completed level: ", average_num_errors_per_completed_level)
			print("Average num actions per completed level: ", average_num_actions_per_completed_level)
				
if __name__ == '__main__':
	get_average_results_per_game(['BoulderDash', 'IceAndFire', 'Catapults'])