from AbstractPlayer import AbstractPlayer
from Types import *

from planning.Translator import Translator
from planning.Planning import Planning

from utils.Types import LEARNING_SSO_TYPE
from planning.parser import Parser

import subprocess
import random
import pickle
import sys


class Agent(AbstractPlayer):
	NUM_GEMS_FOR_EXIT = 9	# For BoulderDash

	def __init__(self):
		"""
		Agent constructor
		Creates a new agent and sets SSO type to JSON
		"""
		AbstractPlayer.__init__(self)
		self.lastSsoType = LEARNING_SSO_TYPE.JSON

		self.config_file = "config/catapults.yaml"
		self.planning = Planning(self.config_file)

		# Game in {'BoulderDash', 'IceAndFire', 'Catapults'}
		self.game_playing = 'Catapults'

	def init(self, sso, elapsedTimer):
		"""
		* Public method to be called at the start of every level of a game.
		* Perform any level-entry initialization here.
		* @param sso Phase Observation of the current game.
		* @param elapsedTimer Timer, which is 1s by default. Modified to 1000s.
							  Check utils/CompetitionParameters.py for more info.
		"""
		self.translator = Translator(sso, self.config_file)

		self.plan = []

	def act(self, sso, elapsedTimer):
		"""
		Method used to determine the next move to be performed by the agent.
		This method can be used to identify the current state of the game and all
		relevant details, then to choose the desired course of action.

		@param sso Observation of the current state of the game to be used in deciding
				   the next action to be taken by the agent.
		@param elapsedTimer Timer, which is 40ms by default. Modified to 500s.
							Check utils/CompetitionParameters.py for more info.
		@return The action to be performed by the agent.
		"""
		
		if len(self.plan) == 0:
			subgoals = self.get_subgoals_positions(sso)
			chosen_subgoal = subgoals[random.randint(0, len(subgoals) - 1)]

			if self.game_playing == 'IceAndFire': # If the game is IceAndFire, check how many types of boots the agent has
				boots_resources = self.get_boots_resources(sso)
			else:
				boots_resources = []

			self.plan = self.search_plan(sso, chosen_subgoal[0], chosen_subgoal[1], boots_resources)

		if len(self.plan) > 0:
			return self.plan.pop(0)
		else:
			return 'ACTION_NIL'


	def search_plan(self, sso, x_goal, y_goal, other_predicates):
		"""
		Method used to search a plan to a goal.

		@param sso State observation.
		@param x_goal X-axis coordinate of the goal to be reached.
		@param y_goal Y-axis coordinate of the goal to be reached.
		@param other_predicates List of predicates to be appended to the ones
								of the current state.
		@return Returns a plan to the current goal.
		"""
		
		# Check if the goal passed as an argument corresponds to the final goal (the exit).
		# In that case, check if its attainable from the current state sso. If not,
		# return an empty plan.

		# Check if the goal corresponds to the exit

		exit = sso.portalsPositions[0][0]
		exit_pos_x = int(exit.position.x // sso.blockSize)
		exit_pos_y = int(exit.position.y // sso.blockSize)

		subgoal_is_final = (exit_pos_x == x_goal and exit_pos_y == y_goal)

		if subgoal_is_final:
			# Check if the final goal is eligible

			if self.game_playing == 'BoulderDash': # The agent needs to have 9 gems 
				keys = sso.avatarResources.keys()

				if len(keys) > 0:
					gem_key = list(sso.avatarResources)[0]
					num_gems = sso.avatarResources[gem_key]
				else:
					num_gems = 0

				if num_gems < self.NUM_GEMS_FOR_EXIT:
					find_plan = False
					plan = []
				else:
					find_plan = True

			elif self.game_playing == 'IceAndFire': # The agent needs to have 10 coins -> there can be no coins on the map
				coin_itype = 10 # Itype of coins

				# Get the number of coins left on the map
				coins_on_the_map = 0

				obs = sso.observationGrid
				X_MAX = sso.observationGridNum
				Y_MAX = sso.observationGridMaxRow

				for y in range(Y_MAX):
					for x in range(X_MAX):
						observation = sso.observationGrid[x][y][0]

						if observation is not None:
							if observation.itype == coin_itype:
								coins_on_the_map += 1

				if coins_on_the_map == 0:
					find_plan = True
				else:
					find_plan = False
					plan = []

			else: # Catapults -> the final goal is always eligible
				find_plan = True
		else:
			find_plan = True # If the goal isn't final, then find a plan

		if find_plan:
			# Find the plan normally
			pddl_predicates, pddl_objects = self.translator.translate_game_state_to_PDDL(sso, other_predicates)
			goal_predicate = self.translator.generate_goal_predicate(x_goal, y_goal)

			self.planning.generate_problem_file(pddl_predicates, pddl_objects, goal_predicate)
			planner_output = self.planning.call_planner()

			plan = self.translator.translate_planner_output(planner_output)

		return plan

	def get_subgoals_positions(self, sso):
		# Note: the final subgoal is always returned, even if it's not eligible!

		if self.game_playing == 'BoulderDash':
			subgoal_pos = [] # Positions of subgoals

			# Final goal
			exit = sso.portalsPositions[0][0]
			exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))
			subgoal_pos.append(exit_pos)
			
			# Gems subgoals
			gems = sso.resourcesPositions[0] 

			for gem in gems:
				gem_x = int(gem.position.x // sso.blockSize) # Convert from pixel to grid positions
				gem_y = int(gem.position.y // sso.blockSize)

				subgoal_pos.append((gem_x, gem_y))

			return subgoal_pos

		elif self.game_playing == 'IceAndFire':
			subgoal_pos = [] # Positions of subgoals

			# Final goal
			exit = sso.portalsPositions[0][0]
			exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))
			subgoal_pos.append(exit_pos)

			# Rest of subgoals

			# Itypes of observations corresponding to subgoals
			subgoals_itypes = (10,9,8) # Itypes of coins, fire boots and ice boots

			obs = sso.observationGrid
			X_MAX = sso.observationGridNum
			Y_MAX = sso.observationGridMaxRow

			for y in range(Y_MAX):
				for x in range(X_MAX):
					observation = sso.observationGrid[x][y][0]

					if observation is not None:
						if observation.itype in subgoals_itypes:
							pos_x = int(observation.position.x // sso.blockSize) # Convert from pixel to grid positions
							pos_y = int(observation.position.y // sso.blockSize) 

							subgoal_pos.append((pos_x, pos_y))

			return subgoal_pos

		elif self.game_playing == 'Catapults':
			# Itypes of observations corresponding to subgoals
			subgoals_itypes = (5,6,7,8) # Itypes of the four different types of catapults

			# Get positions of subgoals
			subgoal_pos = []

			obs = sso.observationGrid
			X_MAX = sso.observationGridNum
			Y_MAX = sso.observationGridMaxRow

			for y in range(Y_MAX):
				for x in range(X_MAX):
					observation = sso.observationGrid[x][y][0]

					if observation is not None:
						if observation.itype in subgoals_itypes:
							pos_x = int(observation.position.x // sso.blockSize) # Convert from pixel to grid positions
							pos_y = int(observation.position.y // sso.blockSize) 

							subgoal_pos.append((pos_x, pos_y))

			# Add the exit position as an elegible subgoal
			exit = sso.portalsPositions[0][0]
			exit_pos_x = int(exit.position.x // sso.blockSize)
			exit_pos_y = int(exit.position.y // sso.blockSize)

			subgoal_pos.append((exit_pos_x, exit_pos_y))

			return subgoal_pos

	# In the IceAndFire game, returns a list with the types of boots the agent has
	def get_boots_resources(self, sso):
		boots_resources = []

		keys = sso.avatarResources.keys()
		ice_boot_key = '8'	# Keys of each boot in the avatarResources dictionary
		fire_boot_key = '9'						

		# Check if the agent has at least one boot	
		if len(keys) > 0:
			if ice_boot_key in keys: # The agent has ice boots
				boots_resources.append("(has-ice-boots)")
			if fire_boot_key in keys: # The agent has fire boots
				boots_resources.append("(has-fire-boots)")

		return boots_resources




