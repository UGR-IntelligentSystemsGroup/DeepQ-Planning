import sys
import subprocess

def parse_plan(in_data, out_plan_file):
    '''
    Function that parses an input plan found by a planner to actions that can
    be read by an agent.

    Args:
        in_data (str): String that contains the output from the planner
        out_plan_file (str): Name of the file where the output will be written        
    '''
    # Boolean that marks when actions are being read
    read_actions = False

    # Write in the output file as the input data is being read
    with open(out_plan_file, 'w') as out_file:
        for line in in_data.split('\n'):
            if 'dig' in line:
                out_file.write('ACTION_USE\n')
            elif 'left' in line:
                out_file.write('ACTION_LEFT\n')
            elif 'right' in line:
                out_file.write('ACTION_RIGHT\n')
            elif 'up' in line:
                out_file.write('ACTION_UP\n')
            elif 'down' in line:
                out_file.write('ACTION_DOWN\n')


def parse_problem(in_description, out_problem):
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
    with open(in_description) as in_file:
        for line in in_file:
            if 'EXIT-LEVEL' in line:
                goals.append('(exited-level player1)\n')
            elif 'GET-GEM' in line:
                get_gems = True
                position = tuple(map(int, line.split(':')[1].split(',')))
                print(position)
                gems_goal.append(position)
            elif 'ORIENTATION' in line:
                player_orientation = line.split(':')[1].replace('\n', '')
                orientation.append('(oriented-{} player1 )\n'.format(player_orientation)) 
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

            # Increase the number of objects found if the element is an object

            # Find out what is the cell's terrain
            if element == 'w':
                terrain = 'wall'
            elif element == '.':
                terrain = 'ground'
            else:
                # If the terrain is empty, check if there is any object in it
                terrain = 'empty'

                if element in game_elements_dict.keys():
                    game_elements_dict[element] += 1
                    
                    n_element = game_elements_dict[element]
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
                    
                    if locatable:
                        positions.append('(at {} {})\n'.format(locatable, cell_str))
                        obj_dict[key].append(locatable)
            
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
    with open(out_problem, 'w') as out:
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


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Error. Program expected 2 more arguments.', file=sys.stderr)
        print('Usage: {} [input problem description filename] [output plan filename]'.format(sys.argv[0]), file=sys.stderr)
        sys.exit(1)

    # Set file names
    command = ['MyAgent/parser/search/search', 'fFlL']
    translate = 'MyAgent/parser/translate/translate.py'
    preprocess = 'MyAgent/parser/preprocess/preprocess'
    domain = 'MyAgent/parser/domain.pddl'
    problem = 'MyAgent/parser/problem.pddl'
    in_file = sys.argv[1]
    out_file = sys.argv[2]

    print('Parsing input problem...')
    parse_problem(in_file, problem)
    print('\nParsing completed!')

    print('Translating domain...\n')
    subprocess.call(['python2', translate, domain, problem])
    print('\nTranslation completed!')

    print('Running preprocessor...\n')
    pre_file = open('output.sas')
    subprocess.call(preprocess, stdin=pre_file)
    pre_file.close()
    print('\nPreprocessing completed!')

    print('Running planner...\n')
    process_file = open('output')
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stdin=process_file)
    out = process.communicate()
    process_file.close()
    print('\nPlanner finished!')

    # Decode output (it is given in bytes)
    output = out[0].decode('ascii')

    print('Parsing output plan...')
    parse_plan(output, out_file)
    print('Parsing completed!')

