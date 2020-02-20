from AbstractPlayer import AbstractPlayer
from Types import *

from utils.Types import LEARNING_SSO_TYPE
from planning.parser import Parser

import random
import numpy as np
import sys

class Agent(AbstractPlayer):
    NUM_GEMS_FOR_EXIT = 9
    DESC_FILE = 'planning/problem.txt'
    OUT_FILE = 'planning/plan.txt'

    def __init__(self):
        """
        Agent constructor
        Creates a new agent and sets SSO type to JSON
        """
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.JSON

        self.current_level = 0 # Level playing right now

        # Set parser
        self.parser = Parser('planning/domain.pddl', 'planning/problem.pddl',
                self.DESC_FILE, self.OUT_FILE)

        # Matrix to save the number of actions used to complete each level
        # num_actions_levels[i] -> actions of level 1
        # num_actions_levels[i][j] -> actions used on iteration j of level i
        self.num_actions_levels = [[],[],[]]

        # Number of iterations (repetitions) for each training level
        self.repetitions = 1

    def init(self, sso, elapsedTimer):
        """
        * Public method to be called at the start of every level of a game.
        * Perform any level-entry initialization here.
        * @param sso Phase Observation of the current game.
        * @param elapsedTimer Timer, which is 1s by default. Modified to 1000s.
                              Check utils/CompetitionParameters.py for more info.
        """

        # Set first turn as True
        self.first_turn = True

        # Create new empty action list
        # This action list corresponds to the plan found by the planner
        self.action_list = []

        # It is true when the agent has the minimum required number of gems
        self.can_exit = False

        # Attribute to save the number of actions used to complete the current level
        self.curr_num_actions = 0


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

        # If the plan is emtpy, get a new one
        if len(self.action_list) == 0:
            # If the agent already has 11 gems, he plans to exit the level
            keys = sso.avatarResources.keys()

            if len(keys) > 0: # He has at least one gem
                gem_key = list(sso.avatarResources)[0] # Python 3

                num_gems = sso.avatarResources[gem_key]

                # Plan to exit the level
                if num_gems >= self.NUM_GEMS_FOR_EXIT:
                    self.can_exit = True

                    exit = sso.portalsPositions[0][0]
                    exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))
 
                    self.action_list = self.search_plan(sso, exit_pos)

                    # Add length of current plan to number of actions used in this level
                    self.curr_num_actions += len(self.action_list)

            # Only plan for next gem if the agent can't exit the level yet
            if not self.can_exit:
                # Obtain gems positions
                gems = self.get_gems_positions(sso)
                
                # <Choose next subgoal>

                avatar_position = (int(sso.avatarPosition[0] // sso.blockSize),
                                 int(sso.avatarPosition[1] // sso.blockSize))

                # Choose a random gem as next goal
                chosen_gem = gems[random.randint(0, len(gems) - 1)]

                # Avoid error of picking a gem which is at the agent's position
                while (chosen_gem[0] == avatar_position[0] and
                       chosen_gem[1] == avatar_position[1]):
                    chosen_gem = gems[random.randint(0, len(gems) - 1)]

                # Search for a plan to chosen_gem
                self.action_list = self.search_plan(sso, chosen_gem)

                # Add length of current plan to number of actions used in this level
                self.curr_num_actions += len(self.action_list)


        # Execute Plan

        # If a plan has been found, return the first action
        if len(self.action_list) > 0:
            return self.action_list.pop(0)
        else:
            print("\n\nPLAN VACIO\n\n")
            return 'ACTION_NIL'


    def get_gems_positions(self, sso):
        # Retrieve gems from current state observation
        gems = sso.resourcesPositions[0] 
        pos = []

        for gem in gems:
            gem_x = int(gem.position.x // sso.blockSize) # Convert from pixel to grid positions
            gem_y = int(gem.position.y // sso.blockSize)

            pos.append((gem_x, gem_y))

        return pos


    def choose_next_subgoal(self, obs_grid, possible_subgoals, avatar_position):
        """
        Uses the learnt model to choose the best subgoal to plan for among all the
        possible subgoals, from the current state of the game.
        The chosen subgoal is the subgoal in "possible_subgoals" which has the smallest
        predicted value by the network. This subgoal can't be in the same position
        than the agent ('avatar_position').

        @param obs_grid Matrix of observations representing the current state of the game
        @param possible_subgoals List of possible subgoals to choose from. Each element
                                 corresponds to a (pos_x, pos_y) pair.
        """

        best_plan_length = 1000000.0
        best_subgoal = (-1, -1)

        for curr_subgoal in possible_subgoals:
            if curr_subgoal[0] != avatar_position[0] or curr_subgoal[1] != avatar_position[1]:
                # Encode input for the network using current state and curr_subgoal
                one_hot_matrix = self.encode_game_state(obs_grid, curr_subgoal)

                # Use the model to predict plan length for curr_subgoal
                plan_length = self.model.predict(one_hot_matrix)

                # See if this is the best plan up to this point
                if plan_length < best_plan_length:
                    best_plan_length = plan_length
                    best_subgoal = curr_subgoal

        return best_subgoal


    def search_plan(self, sso, goal):
        """
        Method used to search for a plan given a description. Writes the resulting
        plan in the output file.

        Change: now returns the plan instead of assigning it to self.action_list

        @param sso State observation.
        @param goal Goal position to be reached. It's given as a (x, y) tuple.        
        """

        action_list = []

        # Parse the current game state
        self.parse_game_state(sso, goal)

        # Call the parser/planner
        self.parser.plan_and_parse()

        # Process the output file to get the plan
        with open(self.OUT_FILE) as plan:
            for action in plan:
                # Remove the \n character at the end and add actions to list
                action_list.append(action.replace('\n', ''))

        return action_list


    def parse_game_state(self, sso, goal):
        """
        Method used to parse a game state. Writes the output in a file to be later
        processed.

        @param sso State observation.
        @param goal Goal position to be reached. It's given as a (x, y) tuple.
        """
        # Get player orientation and parse it
        orientation = sso.avatarOrientation

        if orientation[0] == 1.0:
            orientation_str = 'ORIENTATION:right\n'
        elif orientation[0] == -1.0:
            orientation_str = 'ORIENTATION:left\n'
        elif orientation[1] == 1.0:
            orientation_str = 'ORIENTATION:down\n'
        elif orientation[1] == -1.0:
            orientation_str = 'ORIENTATION:up\n'

        # Get maximum number of rows and columns
        X_MAX = sso.observationGridNum
        Y_MAX = sso.observationGridMaxRow

        # Create a new empty game map
        game_map = []

        # Parse the State observation
        for y in range(Y_MAX):
            # Create new empty row to be written
            row = ''

            for x in range(X_MAX):
                # Get observation
                obs = sso.observationGrid[x][y][0]

                # Process the observation and append it to the row
                if obs == None:
                    row += '-'
                elif obs.itype == 0:
                    row += 'w'
                elif obs.itype == 1:
                    row += 'A'
                elif obs.itype == 4:
                    row += '.'
                elif obs.itype == 5:
                    row += 'e'
                    
                    if (x, y) == goal:
                        goal_str = 'EXIT-LEVEL\n'
                elif obs.itype == 6:
                    row += 'x'
                    
                    if (x, y) == goal:
                        goal_str = 'GET-GEM:{},{}\n'.format(x, y)
                elif obs.itype == 7:
                    row += 'o'
                elif obs.itype == 10:
                    row += 'c'
                elif obs.itype == 11:
                    row += 'b'

            # Add a new line character and add the row to the map
            row += '\n'
            game_map.append(row)


        # Write output in the file
        with open(self.DESC_FILE, 'w') as desc:
            desc.write(orientation_str)
            desc.write(goal_str)
            
            for line in game_map:
                desc.write(line)

        pass
        

    def result(self, sso, elapsedTimer):
        print("Nivel terminado")

        """
        if self.EXECUTION_MODE == 'test' and not self.is_training:
            print("\n\nNúmero de acciones para completar el nivel: {} \n\n".format(self.num_actions_lv))

            # Guardo la suma del número de acciones para completar los dos niveles de validación
            test_output_file = "test_output.txt"

            # No se ha guardado el número de acciones de los dos niveles todavía
            if len(self.num_actions_each_lv) < 2:
                self.num_actions_each_lv.append(self.num_actions_lv) # Guardo el número de acciones del nivel actual

            # Si ya se han completado ambos niveles, guardo las acciones en el fichero y termino la ejecución
            if len(self.num_actions_each_lv) == 2:
                # total_num_actions = self.num_actions_each_lv[0] + self.num_actions_each_lv[1]

                with open(test_output_file, "a") as file:

                    # Imprimo la separación y el nombre del modelo si estamos ejecutando la validación con el primer (el menor) tamaño de dataset
                    if self.num_it_model == self.datasets_sizes_for_training[0]:
                        file.write("\n\n--------------------------\n\n")
                        file.write("Model Name: {}\n\n".format(self.network_name))

                    file.write("{} - level 0 - {}, level 1 - {}\n".format(self.num_it_model, self.num_actions_each_lv[0],
                        self.num_actions_each_lv[1]))

                sys.exit()
        """

        # Save number of actions used to complete this level
        self.num_actions_levels[self.current_level].append(self.curr_num_actions)

        print("\n\n", self.current_level, self.curr_num_actions, "\n\n")

        # After self.repetitions of each level, exit the game
        if len(self.num_actions_levels[2]) >= self.repetitions:

        	print("\n\n-------Number of actions used for each level--------")
        	print("Level 0: ", self.num_actions_levels[0])
        	print("Level 1: ", self.num_actions_levels[1])
        	print("Level 2: ", self.num_actions_levels[2])
        	print("---------------\n\n")

        	# File to save the actions used for each level
        	test_output_file = "num_actions_levels.txt"

        	with open(test_output_file, 'a') as file:
        		file.write("Level 0: {}\n".format(self.num_actions_levels[0]))
        		file.write("Level 1: {}\n".format(self.num_actions_levels[1]))
        		file.write("Level 2: {}\n".format(self.num_actions_levels[2]))

        	sys.exit()


        # Play levels 0-2 in order
        self.current_level = (self.current_level + 1) % 3

        return self.current_level


