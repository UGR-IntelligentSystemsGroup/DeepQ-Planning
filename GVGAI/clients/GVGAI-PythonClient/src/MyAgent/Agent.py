from AbstractPlayer import AbstractPlayer
from Types import *

from utils.Types import LEARNING_SSO_TYPE
# from planning.parser import Parser

import random
import numpy as np
# import tensorflow as tf
# import pickle
import sys

# from LearningModel import CNN

class Agent(AbstractPlayer):
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
        """self.parser = Parser('planning/domain.pddl', 'planning/problem.pddl',
                self.DESC_FILE, self.OUT_FILE)"""


    def init(self, sso, elapsedTimer):
        """
        * Public method to be called at the start of every level of a game.
        * Perform any level-entry initialization here.
        * @param sso Phase Observation of the current game.
        * @param elapsedTimer Timer, which is 1s by default. Modified to 1000s.
                              Check utils/CompetitionParameters.py for more info.
        """

        # Create new empty action list
        # This action list corresponds to the plan found by the planner
        self.action_list = []

        # It is true when the agent has the minimum required number of gems
        # self.can_exit = False

        # See if it's training or validation time
        # self.is_training = not sso.isValidation # It's the opposite to sso.isValidation

        # If it's validation phase, count the number of actions used
        # to beat the current level
        """if self.EXECUTION_MODE == 'test' and not self.is_training:
            self.num_actions_lv = 0"""


    def act(self, sso, elapsedTimer):
       	return 'ACTION_NIL'


    def get_gems_positions(self, sso):
        # Retrieve gems from current state observation
        
        """gems = sso.resourcesPositions[0] 
        pos = []

        for gem in gems:
            gem_x = int(gem.position.x // sso.blockSize) # Convert from pixel to grid positions
            gem_y = int(gem.position.y // sso.blockSize)

            pos.append((gem_x, gem_y))

        return pos"""

    def encode_game_state(self, obs_grid, goal_pos):
        """
        Transforms the game state (sso) from a matrix of observations to a matrix in which each
        position is one-hot encoded. If there are more than one observations at the same position,
        both are encoded. If there are no observations at a position, the resulting encoding
        is an array full of zeroes

        @param obs_grid Matrix of observations
        @param goal_pos (x, y) position (not as pixels but as grid) of the selected goal  
        """

        # 0, 1, 4, 5, 6, 7, 10, 11

        # Dictionary that maps itype (key) to one-hot array position to write a 1
        # e.g. 4 : 2 = [0 0 1 0 0 0 0 0 0]
        # None (empty tiles) objects are assigned an array full of zeroes
        encode_dict = {
            0 : 0,
            1 : 1,
            4 : 2,
            5 : 3,
            6 : 4,
            7 : 5,
            10 : 6,
            11 : 7
        }

        num_cols = len(obs_grid)
        num_rows = len(obs_grid[0])

        one_hot_length = 8 + 1 # 1 extra position to represent the objective (the last 1)

        # The image representation is by rows and columns, instead of (x, y) pos of each pixel
        # Row -> y
        # Col -> x
        one_hot_grid = np.zeros((num_rows, num_cols, one_hot_length), np.int8)

        # Encode the grid
        for x in range(num_cols):
            for y in range(num_rows):
                for obs in obs_grid[x][y]:
                    if obs is not None and obs.itype != 3: # Ignore Empty Tiles and pickage images (those that correspond to ACTION_USE)
                        this_pos = encode_dict[obs.itype]
                        one_hot_grid[y][x][this_pos] = 1

        # Encode the goal position within that grid
        one_hot_grid[goal_pos[1]][goal_pos[0]][one_hot_length-1] = 1

        return one_hot_grid


    def search_plan(self, sso, goal):
        """<TODO>"""


    def parse_game_state(self, sso, goal):
        """<TODO>"""

    def result(self, sso, elapsedTimer):
        print("Nivel terminado")

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


        # Play levels 0-2 in order
        self.current_level = (self.current_level + 1) % 3

        return self.current_level