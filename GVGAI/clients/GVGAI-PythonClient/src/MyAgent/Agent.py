from AbstractPlayer import AbstractPlayer
from Types import *

from utils.Types import LEARNING_SSO_TYPE

import subprocess
import random

class Agent(AbstractPlayer):
    NUM_GEMS_FOR_EXIT = 9

    def __init__(self):
        """
        Agent constructor
        Creates a new agent and sets SSO type to JSON
        """
        AbstractPlayer.__init__(self)
        self.lastSsoType = LEARNING_SSO_TYPE.JSON


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
            # Set level description and output file names
            out_description = 'MyAgent/problem.txt'
            output_file = 'MyAgent/out.txt'

            # If the agent already has 11 gems, he plans to exit the level
            keys = sso.avatarResources.keys()

            if len(keys) > 0: # He has at least one gem
                gem_key = keys[0]
                num_gems = sso.avatarResources[gem_key]

                print(num_gems)

                if num_gems >= self.NUM_GEMS_FOR_EXIT:
                    # Plan to exit the level
                    print("YA PUEDO SALIR DEL NIVEL")
                    exit = sso.portalsPositions[0][0]
                    exit_pos = (exit.position.x // sso.blockSize, exit.position.y // sso.blockSize)

                    self.search_plan(sso, exit_pos, out_description, output_file)


            # Obtain gems positions
            gems = self.get_gems_positions(sso)
            
            chosen_gem = gems[random.randint(0, len(gems) - 1)] # Choose a random gem as next objective

            # Search for a plan
            self.search_plan(sso, chosen_gem, out_description, output_file)

        # If a plan has been found, return the first action
        if len(self.action_list) > 0:
            return self.action_list.pop(0)
        else:
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

    def search_plan(self, sso, goal, description, output):
        """
        Method used to search for a plan given a description. Writes the resulting
        plan in the output file.

        @param sso State observation.
        @param goal Goal position to be reached. It's given as a (x, y) tuple.        
        @param description Name of the file containing the description of the current
                           game state.
        @param output Name of the file which will contain the output plan
        """
        # Parse the current game state
        self.parse_game_state(sso, goal, description)

        # Call the subprocess that searchs for a plan
        subprocess.call(['python3', 'MyAgent/parser/parser.py', description, output])

        # Process the output file to get the plan
        with open(output) as plan:
            for action in plan:
                # Remove the \n character at the end and add actions to list
                self.action_list.append(action.replace('\n', ''))


    def parse_game_state(self, sso, goal, description):
        """
        Method used to parse a game state. Writes the output in a file to be later
        processed.

        @param sso State observation.
        @param goal Goal position to be reached. It's given as a (x, y) tuple.
        @param description Name of the file where the output will be written.
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
        with open(description, 'w') as desc:
            desc.write(orientation_str)
            desc.write(goal_str)
            
            for line in game_map:
                desc.write(line)

        pass


