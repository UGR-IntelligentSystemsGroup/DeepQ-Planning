"""
This script parses the execution results stored in test_output.txt
"""

import re
import numpy as np



import sys

output_file = "test_output.txt"
# games = ['BoulderDash', 'IceAndFire', 'Catapults']
games = ['BoulderDash', 'IceAndFire', 'Catapults']
num_test_levels = 11 # Number of levels each model is tested on (11 = 5 default test levels + 6 new (hard) test levels)
model = "DQP" # 'DQP' o 'Random'

def get_average_results_per_game(game_list, num_test_levels=5, model="DQP"):
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

		if model == "DQP":
			# Group 1 -> Name of the network
			# Group 2 -> Number of levels it was trained on
			match_list = re.findall(r'(DQN.*)_[0-9]+-([0-9]+) \| {} \|.*'.format(curr_game), results_str)
		else:
			# Group 1 -> Name of the network
			match_list = re.findall(r'(DQN.*)_[0-9]+ \| {} \|.*'.format(curr_game), results_str)


		# Convert it to a set to get the unique values (model names)
		match_set = set(match_list)



		# For each different model of the current game, match the corresponding test/validation results
		for curr_match in match_set:
			# Get all the lines containing the results
			if model == "DQP":
				curr_model_match_list = re.findall(r'{}_[0-9]+-{} \| {} \| (-?[0-9]+) \| ([0-9]+) \| (\d+.?\d*) \| (\d+.?\d*)'.
					format(curr_match[0], curr_match[1], curr_game), results_str)

				# Separe the number of errors, actions, planning and goal selection time
				curr_model_num_errors = [int(match[0]) for match in curr_model_match_list]
				curr_model_num_actions = [int(match[1]) for match in curr_model_match_list]
				curr_model_planning_time = [float(match[2]) for match in curr_model_match_list]
				curr_model_goal_selec_time = [float(match[3]) for match in curr_model_match_list]
			else:
				curr_model_match_list = re.findall(r'{}_[0-9]+ \| {} \| (-?[0-9]+) \| ([0-9]+) \| (\d+.?\d*)'.
					format(curr_match, curr_game), results_str)

				# Separe the number of errors, actions, planning and goal selection time
				curr_model_num_errors = [int(match[0]) for match in curr_model_match_list]
				curr_model_num_actions = [int(match[1]) for match in curr_model_match_list]
				curr_model_planning_time = [float(match[2]) for match in curr_model_match_list]
				curr_model_goal_selec_time = [0 for match in curr_model_match_list] # The Random Model does not spend time selecting subgoals


			#print(len(curr_model_num_errors))
			#sys.exit()

			if curr_game in ('BoulderDash', 'IceAndFire'):
				# Calculate the average and std number of errors and actions for each of the
				# five test/validation levels
				average_num_errors_per_level = [np.average(curr_model_num_errors[i::num_test_levels]) for i in range(num_test_levels)]
				average_num_actions_per_level = [np.average(curr_model_num_actions[i::num_test_levels]) for i in range(num_test_levels)]
				std_num_errors_per_level = [np.std(curr_model_num_errors[i::num_test_levels]) for i in range(num_test_levels)]
				std_num_actions_per_level = [np.std(curr_model_num_actions[i::num_test_levels]) for i in range(num_test_levels)]

				# Calculate the average and std total time and mean time for each of the five
				# test/validation levels
				average_planning_time_per_level = [np.average(curr_model_planning_time[i::num_test_levels]) for i in range(num_test_levels)]
				average_goal_selec_time_per_level = [np.average(curr_model_goal_selec_time[i::num_test_levels]) for i in range(num_test_levels)]
				std_planning_time_per_level = [np.std(curr_model_planning_time[i::num_test_levels]) for i in range(num_test_levels)]
				std_goal_selec_time_per_level = [np.std(curr_model_goal_selec_time[i::num_test_levels]) for i in range(num_test_levels)]

				print("\n--- {} ---\n".format(curr_game))

				if model == "DQP":
					print("{} - lvs={}".format(curr_match[0], curr_match[1])) # Model Name and number of levels
				else:
					print("{}".format(curr_match)) # Model Name
				print("\nAverage num errors per level: ", average_num_errors_per_level)
				print("Std num errors per level: ", std_num_errors_per_level)
				print("\nAverage num actions per level: ", average_num_actions_per_level)
				print("Std num actions per level: ", std_num_actions_per_level)
				print("\nAverage planning time per level: ", average_planning_time_per_level)
				print("Std planning time per level: ", std_planning_time_per_level)
				print("\nAverage goal selection time per level: ", average_goal_selec_time_per_level)
				print("Std goal selection time per level: ", std_goal_selec_time_per_level)

			else: # Catapults
				# Calculate success_rate (percentage of levels the agent beats)



				success_rate_per_level = [(len(curr_model_num_errors[i::num_test_levels]) - list(curr_model_num_errors[i::num_test_levels]).count(-1))
				 / len(curr_model_num_errors[i::num_test_levels]) * 100 for i in range(num_test_levels)]

				
				# Calculate the average and std number of errors per test level <completed>
				average_num_errors_per_completed_level = []
				std_num_errors_per_completed_level = []
				for i in range(num_test_levels):
					# Obtain the number of errors of the current level
					curr_num_errors = curr_model_num_errors[i::num_test_levels]

					# Delete the -1 entries (only use the errors of the completed levels)
					curr_num_errors = [e for e in curr_num_errors if e != -1]

					# If no level was completed, the average and std number of errors is -1 (undetermined)
					if len(curr_num_errors) == 0:
						average_num_errors_per_completed_level.append(-1)
						std_num_errors_per_completed_level.append(-1)
					else: # otherwise, we calculate the average and std
						average_num_errors_per_completed_level.append(np.average(curr_num_errors))
						std_num_errors_per_completed_level.append(np.std(curr_num_errors))

				# Calculate the average and std number of actions, average planning time and average goal selec time
				# per test level <completed>
				average_num_actions_per_completed_level = []
				average_planning_time_per_completed_level = []
				average_goal_selec_time_per_completed_level = []
				std_num_actions_per_completed_level = []
				std_planning_time_per_completed_level = []
				std_goal_selec_time_per_completed_level = []

				for i in range(num_test_levels):
					# Obtain the number of errors, planning time and goal selec time of the current level
					curr_num_actions = curr_model_num_actions[i::num_test_levels]
					curr_planning_time = curr_model_planning_time[i::num_test_levels]
					curr_goal_selec_time = curr_model_goal_selec_time[i::num_test_levels]

					# Delete the entries of levels with -1 errors (levels not completed)
					curr_num_errors = curr_model_num_errors[i::num_test_levels]
					curr_num_actions = [a for a,e in zip(curr_num_actions, curr_num_errors) if e != -1]
					curr_planning_time = [a for a,e in zip(curr_planning_time, curr_num_errors) if e != -1]
					curr_goal_selec_time = [a for a,e in zip(curr_goal_selec_time, curr_num_errors) if e != -1]

					# If no level was completed, the average number of actions, total time and
					# mean time is -1 (undetermined)
					if len(curr_num_actions) == 0:
						average_num_actions_per_completed_level.append(-1)
						average_planning_time_per_completed_level.append(-1)
						average_goal_selec_time_per_completed_level.append(-1)
						std_num_actions_per_completed_level.append(-1)
						std_planning_time_per_completed_level.append(-1)
						std_goal_selec_time_per_completed_level.append(-1)
					else: # otherwise, we calculate the average and std
						average_num_actions_per_completed_level.append(np.average(curr_num_actions))
						average_planning_time_per_completed_level.append(np.average(curr_planning_time))
						average_goal_selec_time_per_completed_level.append(np.average(curr_goal_selec_time))
						std_num_actions_per_completed_level.append(np.std(curr_num_actions))
						std_planning_time_per_completed_level.append(np.std(curr_planning_time))
						std_goal_selec_time_per_completed_level.append(np.std(curr_goal_selec_time))


				print("\n--- {} ---\n".format(curr_game))
				if model == "DQP":
					print("{} - lvs={}".format(curr_match[0], curr_match[1])) # Model Name and number of levels
				else:
					print("{}".format(curr_match)) # Model Name
				print("Success rate per level", success_rate_per_level)
				print("\nAverage num errors per completed level: ", average_num_errors_per_completed_level)
				print("Std num errors per completed level: ", std_num_errors_per_completed_level)
				print("\nAverage num actions per completed level: ", average_num_actions_per_completed_level)
				print("Std num actions per completed level: ", std_num_actions_per_completed_level)
				print("\nAverage planning time per completed level: ", average_planning_time_per_completed_level)
				print("Std planning time per completed level: ", std_planning_time_per_completed_level)
				print("\nAverage goal selection time per completed level: ", average_goal_selec_time_per_completed_level)
				print("Std goal selection time per completed level: ", std_goal_selec_time_per_completed_level)

if __name__ == '__main__':
	get_average_results_per_game(games, num_test_levels, model)