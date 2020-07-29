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
	NUM_GEMS_FOR_EXIT = 9

	def __init__(self):
		"""
		Agent constructor
		Creates a new agent and sets SSO type to JSON
		"""
		AbstractPlayer.__init__(self)
		self.lastSsoType = LEARNING_SSO_TYPE.JSON

		self.config_file = "config/ice-and-fire.yaml"
		self.planning = Planning(self.config_file)

		# Game in {'BoulderDash', 'IceAndFire', 'Catapults'}
		self.game_playing = 'IceAndFire'

	def init(self, sso, elapsedTimer):
		"""
		* Public method to be called at the start of every level of a game.
		* Perform any level-entry initialization here.
		* @param sso Phase Observation of the current game.
		* @param elapsedTimer Timer, which is 1s by default. Modified to 1000s.
							  Check utils/CompetitionParameters.py for more info.
		"""
		self.translator = Translator(sso, self.config_file)

		self.plan = self.search_plan(sso, 14, 12, [])

		# QUITAR!!!!!
		self.var = True

		# Keys de los recursos de IceAndFire (sso.avatarResources)
		# '8' -> Bota de nieve

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

		# ["(has-ice-boots)", "(has-fire-boots)"]
	
		self.get_boots_resources(sso)

		if len(self.plan) == 0 and self.var:
			"""subgoals = self.get_subgoals_positions(sso)
			chosen_subgoal = subgoals[random.randint(0, len(subgoals) - 1)]

			self.plan = self.search_plan(sso, chosen_subgoal[0], chosen_subgoal[1], [])"""

			self.plan = self.search_plan(sso, 4, 3, ["(has-ice-boots)"]) # Posición de la bota de fuego

			print(self.plan)

			self.var = False


			pass


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
		pddl_predicates, pddl_objects = self.translator.translate_game_state_to_PDDL(sso, other_predicates)
		goal_predicate = self.translator.generate_goal_predicate(x_goal, y_goal)

		self.planning.generate_problem_file(pddl_predicates, pddl_objects, goal_predicate)
		planner_output = self.planning.call_planner()

		plan = self.translator.translate_planner_output(planner_output)

		return plan

	def get_subgoals_positions(self, sso):

		if self.game_playing == 'BoulderDash':
			# Check if the agent has already got 9 gems
			keys = sso.avatarResources.keys()

			# Check if the agent has at least one gem
			if len(keys) > 0:
				gem_key = list(sso.avatarResources)[0]
				num_gems = sso.avatarResources[gem_key]
			else:
				num_gems = 0

			# Return the final goal (the position of the exit)
			if num_gems >= self.NUM_GEMS_FOR_EXIT:
				# self.can_exit = True

				exit = sso.portalsPositions[0][0]
				exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))

				return [exit_pos]
			
			else: # Return gems positions
				# Retrieve gems from current state observation
				gems = sso.resourcesPositions[0] 
				pos = []

				for gem in gems:
					gem_x = int(gem.position.x // sso.blockSize) # Convert from pixel to grid positions
					gem_y = int(gem.position.y // sso.blockSize)

					pos.append((gem_x, gem_y))

				return pos

		elif self.game_playing == 'IceAndFire':
			# Itypes of observations corresponding to subgoals
			subgoals_itypes = (10,9,8) # Itypes of coins, fire boots and ice boots
			coin_itype = 10

			# Get positions of subgoals and number of coins on the map
			subgoal_pos = []
			n_coins = 0

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

							# If the subgoal is a coin, count it
							if observation.itype == coin_itype:
								n_coins += 1

			# If there are no coins on the map, the subgoal corresponds to the exit
			if n_coins == 0:
				# self.can_exit = True
				exit = sso.portalsPositions[0][0]
				exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))

				return [exit_pos]

			else:
				return subgoal_pos

		elif self.game_playing == 'Catapults':
			pass

	# In the IceAndFire game, returns a list with the types of boots the agent has
	def get_boots_resources(self, sso):
		keys = sso.avatarResources.keys()

		# Check if the agent has at least one boot
		if len(keys) > 0:
			gem_key = list(sso.avatarResources)[0]
			print(sso.avatarResources)
		else:
			print("No tiene botas")




