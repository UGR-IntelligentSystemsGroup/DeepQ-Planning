import sys
import subprocess
import time

class Parser:
    def __init__(self, domain, problem, in_desc, out_plan):
        # Set file names
        self.domain = domain
        self.problem = problem
        self.in_desc = in_desc
        self.out_plan = out_plan

        # Set planner
        self.planner = ['planning/ff', '-o', domain, '-f', problem, '-O', '-g', '1', '-h', '1']


    def parse_plan(self, in_data):
        '''
        Function that parses an input plan found by a planner to actions that can
        be read by an agent.

        Args:
            in_data (str): String that contains the output from the planner
        '''
        # Boolean that marks when actions are being read
        read_actions = False

        # Write in the output file as the input data is being read
        with open(self.out_plan, 'w') as out_file:
            for line in in_data.split('\n'):
                if 'dig' in line.lower():
                    out_file.write('ACTION_USE\n')
                elif 'left' in line.lower():
                    out_file.write('ACTION_LEFT\n')
                elif 'right' in line.lower():
                    out_file.write('ACTION_RIGHT\n')
                elif 'up' in line.lower():
                    out_file.write('ACTION_UP\n')
                elif 'down' in line.lower():
                    out_file.write('ACTION_DOWN\n')


    def parse_problem(self):
        '''
        Function that parses an input problem described in a plain text file and
        transforms the description to a .pddl file which uses a certain domain.

        Args:
            in_description (str): Name of the file where the problem is described
            out_problem (str): Name of the file where the .pddl problem will be written
        '''
        # Set the game map as a new empty list
        game_map = []

        # Create an empty list of goal geams
        gems_goal = []

        # Set the initial goals list
        goals = []

        # Boolean that indicates wether gems are involved in the goal state or not
        get_gems = False

        # Set initial lists for init
        terrains = []
        connections = []
        positions = []
        orientation = []

        # Set initial dictionary for objects
        obj_dict = {'Player': [], 'Gem': [], 'Boulder': [], 'Cell': [], 'Exit': [],
                    'Bat': [], 'Scorpion': []
                }

        # Set the tab increment
        tab_increment = 4

        # Set each number of elements of the game to 0
        # Each symbol represents the following:
            # . > Ground terrain
            # - > Empty terrain
            # e > Exit
            # o > Boulder
            # x > Gem
            # c > Scorpion enemy
            # b > Bat enemy
            # A > Player
        # . and - are not used because there are no objects associated to those
        # elements
        game_elements_dict = {'e': 0, 'o': 0, 'x': 0, 'c': 0, 'b': 0, 'A': 0}

        # Process input map file
        with open(self.in_desc) as in_file:
            for line in in_file:
                if 'EXIT-LEVEL' in line:
                    goals.append('(exited-level player1)\n')
                    print('Goal: exit the current level.')
                elif 'GET-GEM' in line:
                    get_gems = True
                    position = tuple(map(int, line.split(':')[1].split(',')))
                    print('Goal: get gem at {}'.format(position))
                    gems_goal.append(position)
                elif 'ORIENTATION' in line:
                    player_orientation = line.split(':')[1].replace('\n', '')
                    orientation.append('(oriented-{} player1)\n'.format(player_orientation)) 
                elif not line == '\n':
                    game_map.append(list(line.replace('\n', '')))

        # Get the number of rows and columns using the map size
        rows = len(game_map)
        columns = len(game_map[0])
        print('Processing a map with {} rows and {} columns'.format(rows, columns))

        # Process the map
        for x in range(rows):
            for y in range(columns):
                # Get the current element and cell
                element = game_map[x][y]
                cell_str = 'c_{}_{}'.format(y, x)

                # Find out what is the cell's terrain
                if element == 'w':
                    terrain = 'wall'
                elif element == '.':
                    terrain = 'ground'
                else:
                    # If the terrain is empty, check if there is any object in it
                    terrain = 'empty'

                    if element in game_elements_dict.keys():
                        # Increase the number of objects found if the element is an object
                        game_elements_dict[element] += 1
                        
                        # Get the number of elements
                        n_element = game_elements_dict[element]

                        # Initially assume that the cell is empty
                        locatable = None
                        
                        # Check what object is in the cell, if any
                        if element == 'e':
                            locatable = 'exit{}'.format(n_element)
                            key = 'Exit'
                        elif element == 'o':
                            locatable = 'boulder{}'.format(n_element)
                            key = 'Boulder'
                        elif element == 'x':
                            locatable = 'gem{}'.format(n_element)
                            key = 'Gem'
                            
                            # If the gem is one of the goal gems, add it to the goal list
                            if get_gems and (y, x) in gems_goal:
                                goals.append('(got {})\n'.format(locatable))
                        elif element == 'c':
                            locatable = 'scorpion{}'.format(n_element)
                            key = 'Scorpion'
                        elif element == 'b':
                            locatable = 'bat{}'.format(n_element)
                            key = 'Bat'
                        elif element == 'A':
                            locatable = 'player{}'.format(n_element)
                            key = 'Player'
                        
                        # If any locatable has been found, add it to the position list
                        if locatable:
                            positions.append('(at {} {})\n'.format(locatable, cell_str))
                            obj_dict[key].append(locatable)
                
                # Add terrain to the terrain list
                terrains.append('(terrain-{} {})\n'.format(terrain, cell_str))
                obj_dict['Cell'].append(cell_str)

                # Find out which connections must be added to the list
                if x - 1 >= 0:
                    connections.append('(connected-up {} c_{}_{})\n'.format(cell_str, y, x - 1))
                if x + 1 < rows:
                    connections.append('(connected-down {} c_{}_{})\n'.format(cell_str, y, x + 1))
                if y + 1 < columns:
                    connections.append('(connected-right {} c_{}_{})\n'.format(cell_str, y + 1, x))
                if y - 1 >= 0:
                    connections.append('(connected-left {} c_{}_{})\n'.format(cell_str, y - 1, x))

        # Write output
        with open(self.problem, 'w') as out:
            # Set tab-size
            tab_size = tab_increment

            # Write problem line
            out.write('(define (problem Boulders)\n')

            # Write domain
            out.write('\t(:domain Boulderdash)\n'.expandtabs(tab_size))

            # Write objects
            out.write('\t(:objects\n'.expandtabs(tab_size))
            tab_size += tab_increment

            for key in obj_dict.keys():
                if len(obj_dict[key]):
                    elements = ' '.join(obj_dict[key])
                    out.write('\t{} - {}\n'.expandtabs(tab_size).format(elements, key))

            tab_size -= tab_increment
            out.write('\t)\n'.expandtabs(tab_size))

            # Write init
            init = terrains + connections + positions + orientation
            out.write('\t(:init\n'.expandtabs(tab_size))
            tab_size += tab_increment

            for i in init:
                out.write('\t{}'.expandtabs(tab_size).format(i))

            tab_size -= tab_increment
            out.write('\t)\n'.expandtabs(tab_size))

            # Write goal
            out.write('\t(:goal\n'.expandtabs(tab_size))
            tab_size += tab_increment

            out.write('\t(AND\n'.expandtabs(tab_size))
            tab_size += tab_increment

            for g in goals:
                out.write('\t{}'.expandtabs(tab_size).format(g))

            tab_size -= tab_increment
            out.write('\t)\n'.expandtabs(tab_size))

            tab_size -= tab_increment
            out.write('\t)\n'.expandtabs(tab_size))

            out.write(')')


    def plan_and_parse(self):
        # Get initial time
        time_ini = time.time()

        # Parse the input problem
        print('Parsing input problem...')
        self.parse_problem()
        print('Parsing completed in {} seconds!'.format(time.time() - time_ini))

        # Run the planer
        print('\nRunning planner...')
        plan_time = time.time()
        process = subprocess.Popen(self.planner, stdout=subprocess.PIPE)
        out = process.communicate()
        print('Planner finished in {} seconds!'.format(time.time() - plan_time))

        # Decode output (it is given in bytes)
        output = out[0].decode('ascii')

        # Parse the output plan
        print('\nParsing output plan...')
        out_parse_time = time.time()
        self.parse_plan(output)
        print('Parsing completed in {} seconds!'.format(time.time() - out_parse_time))

        time_end = time.time()
        print('\nTotal time: {} seconds'.format(time_end - time_ini))
