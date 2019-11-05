from AbstractPlayer import AbstractPlayer
from Types import *

from utils.Types import LEARNING_SSO_TYPE
from planning.parser import Parser

import subprocess
import random
import numpy as np
import tensorflow as tf
import pickle
import sys

from LearningModel import DQNetwork

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


        # << Choose execution mode >>
        # - 'create_dataset' -> It doesn't train any model. Just creates the dataset (experience replay) and saves it
        # - 'train' -> It loads the saved dataset, trains the model with it, and saves the trained model. 
        #              It doesn't add any sample to the experience replay
        # - 'test' -> It loads the trained model and tests it on the validation levels, obtaining the metrics.


        self.EXECUTION_MODE="test" # Automatically changed by ejecutar_pruebas.py!

        # Name of the DQNetwork. Also used for creating the name of file to save and load the model from
        self.network_name="DQN_alfa-0.005_dropout-0.4_batch-16_its-5000_27" # Automatically changed by ejecutar_pruebas.py!

        # Sizes of datasets to train the model on. For each size, a different model is created and trained in the training phase.
        self.datasets_sizes_for_training = [500, 1000, 2500, 5000, 7500, 10000]

        # <TODO>
        # Cambiar de random sampling del experience replay a secuencial (tras aleatorizar el vector con np.shuffle)

        if self.EXECUTION_MODE == 'create_dataset':

            # Attribute that stores agent's experience to implement experience replay (for training)
            # Each element is a tuple (s, r, s') corresponding to:
            # - s -> start_state and chosen subgoal ((game state, chosen subgoal) one-hot encoded). Shape = (-1, 13, 26, 9).
            # - r -> length of the plan from the start_state to the chosen subgoal.
            # - s' -> state of the game just after the agent has achieved the chosen subgoal.
            #         It's an instance of the SerializableStateObservation (sso) class.
            self.memory = []

            # Path of the file to save the experience replay to
            self.dataset_save_path = 'SavedDatasets/' + 'dataset_1000_15.dat'

            # Size of the experience replay to save. When the number of samples reaches this number, the experience replay (self.memory)
            # is saved and the program exits
            self.num_samples_for_saving_dataset = 1000

        elif self.EXECUTION_MODE == 'train':
            # Parameters of the Learning Model
            # Automatically changed by ejecutar_pruebas.py!
            self.learning_rate=0.005
            self.dropout_prob=0.4
            self.num_train_its=5000
            self.batch_size=16
            
            self.max_tau = 250 # Number of training its before copying the DQNetwork's weights to the target network
            self.tau = 0 # Counter that resets to 0 when the target network is updated
            self.gamma = 0.9 # Discount rate for Deep Q-Learning

            # Name of the saved model file (without the number of dataset size part)
            self.model_save_path = "./SavedModels/" + self.network_name + ".ckpt"

            # Experience Replay
            self.memory = [] # Attribute to save the dataset

            # Path of the dataset to load
            self.dataset_load_path = 'SavedDatasets/' + 'dataset_1000' # Without the '_1.dat' part

        else: # Test
            # Create Learning Model

            # DQNetwork
            self.model = DQNetwork(writer_name=self.network_name,
                     l1_num_filt = 4, l1_window = [4,4], l1_strides = [2,2],
                     padding_type = "SAME",
                     max_pool_size = [2, 2],
                     max_pool_str = [1, 1],
                     fc_num_units = [64, 16], dropout_prob = 0.6,
                     learning_rate = 0.005)


            # Name of the saved model file to load (without the number of training steps part)
            model_load_path = "./SavedModels/" + self.network_name + ".ckpt"

            # Number of iterations of the model to load
            # Automatically changed by ejecutar_pruebas.py!
            self.num_it_model=10000

            # Array to save the number of actions used to complete each level to save it to the output file
            self.num_actions_each_lv = []

            # <Load the already-trained model in order to test performance>
            self.model.load_model(path = model_load_path, num_it = self.num_it_model)


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

        # See if it's training or validation time
        self.is_training = not sso.isValidation # It's the opposite to sso.isValidation

        # If it's validation phase, count the number of actions used
        # to beat the current level
        if self.EXECUTION_MODE == 'test' and not self.is_training:
            self.num_actions_lv = 0


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

        # <Train the model without playing the game (EXECUTION_MODE == 'train')>

        if self.EXECUTION_MODE == 'train':

            # Train a different model for each different dataset size
            for dataset_size in self.datasets_sizes_for_training:
                # Load dataset of current size
                self.load_dataset(self.dataset_load_path, num_elements=dataset_size)

                # Shuffle dataset
                random.shuffle(self.memory)

                # Create Learning model

                curr_name = self.network_name + "_{}".format(dataset_size) # Append dataset size to the name of the network
                tf.reset_default_graph() # Clear Tensorflow Graph and Variables

                # DQNetwork
                self.model = DQNetwork(writer_name=curr_name,
                         l1_num_filt = 4, l1_window = [4,4], l1_strides = [2,2],
                         padding_type = "SAME",
                         max_pool_size = [2, 2],
                         max_pool_str = [1, 1],
                         fc_num_units = [64, 16], dropout_prob = self.dropout_prob,
                         learning_rate = self.learning_rate)

                # Target Network
                # Used to predict the Q targets. It is upgraded every max_tau updates.
                self.target_network = DQNetwork(name="TargetNetwork",
                         create_writer = False,
                         l1_num_filt = 4, l1_window = [4,4], l1_strides = [2,2],
                         padding_type = "SAME",
                         max_pool_size = [2, 2],
                         max_pool_str = [1, 1],
                         fc_num_units = [64, 16], dropout_prob = 0.0,
                         learning_rate = self.learning_rate)

                # Initialize target network's weights with those of the DQNetwork
                self.update_target_network()
                self.tau = 0

                num_samples = len(self.memory)

                print("\n> Started training of model with dataset size={}\n".format(dataset_size))

                ind_batch = 0 # Index for selecting the next minibatch

                # Execute the training of the current model
                for curr_it in range(self.num_train_its):   
                    # Choose next batch from Experience Replay

                    if ind_batch+self.batch_size < num_samples:
                        batch = self.memory[ind_batch:ind_batch+self.batch_size]
                        ind_batch = (ind_batch + self.batch_size)
                    else: # Got to the end of the experience replay -> shuffle it and start again
                        batch = self.memory[ind_batch:]
                        ind_batch = 0

                        random.shuffle(self.memory)

                    batch_X = np.array([each[0] for each in batch]) # inputs for the DQNetwork
                    batch_R = [each[1] for each in batch] # r values (plan lenghts)
                    batch_S = [each[2] for each in batch] # s' values (sso instances)

                    # Calculate Q_targets
                    Q_targets = []
                    
                    for r, s in zip(batch_R, batch_S):
                        Q_target = r + self.gamma*self.get_min_Q_value(s)
                        Q_targets.append(Q_target)

                    Q_targets = np.reshape(Q_targets, (-1, 1)) 

                    # Execute one training step
                    self.model.train(batch_X, Q_targets)
                    self.tau += 1

                    # Update target network every tau training steps
                    if self.tau >= self.max_tau:
                        update_ops = self.update_target_network()
                        self.target_network.update_weights(update_ops)

                        self.tau = 0

                    # Save Logs
                    self.model.save_logs(batch_X, Q_targets, curr_it)

                    # Periodically print the progress of the training
                    if curr_it % 500 == 0 and curr_it != 0:
                        print("- {} its completed".format(curr_it))

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
                # Save the current trained model
                self.model.save_model(path = self.model_save_path, num_it = dataset_size)
                print("\n> Current model saved! Dataset size={}\n".format(dataset_size))


            # Exit the program after finishing training
            print("\nTraining finished!")
            sys.exit()
  

        # <Play the game (EXECUTION_MODE == 'create_dataset' or 'test')>

        if self.EXECUTION_MODE == 'create_dataset':
            print("\nExperience Replay size: {}\n".format(len(self.memory)))

        # If the plan is emtpy, get a new one
        if len(self.action_list) == 0:
            # If the agent already has 11 gems, he plans to exit the level
            keys = sso.avatarResources.keys()

            if len(keys) > 0: # He has at least one gem
                # gem_key = keys[0] # Python 2
                gem_key = list(sso.avatarResources)[0] # Python 3

                num_gems = sso.avatarResources[gem_key]

                # Plan to exit the level
                if num_gems >= self.NUM_GEMS_FOR_EXIT:
                    self.can_exit = True

                    exit = sso.portalsPositions[0][0]
                    exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))
 
                    self.action_list = self.search_plan(sso, exit_pos)

                    # Add sample to memory
                    if self.EXECUTION_MODE == 'create_dataset' and self.is_training:
                        # Save current observation for dataset
                        one_hot_grid = self.encode_game_state(sso.observationGrid, exit_pos)

                        # Save plan length as current metric
                        plan_metric = len(self.action_list)

                        # Add the current state s' to self.mem_sample, 
                        # create a new self.mem_sample and add to it s, r and s' (None state, corresponding to finished level)
                        self.mem_sample[2] = sso

                        # Add old mem_sample to memory
                        self.memory.append(self.mem_sample)

                        # Create new mem_sample and append it to memory
                        self.mem_sample = [one_hot_grid, plan_metric, None]
                        self.memory.append(self.mem_sample)


            # Only plan for next gem if the agent can't exit the level yet
            if not self.can_exit:
                # Obtain gems positions
                gems = self.get_gems_positions(sso)
                
                # <Choose next subgoal>

                avatar_position = (int(sso.avatarPosition[0] // sso.blockSize),
                                 int(sso.avatarPosition[1] // sso.blockSize))

                if self.is_training: # Choose a random gem if the agent is training
                    chosen_gem = gems[random.randint(0, len(gems) - 1)] # Choose a random gem as next objective
                else: # Choose best gem if the agent is on validation time
                    chosen_gem = self.choose_next_subgoal(sso.observationGrid, gems, avatar_position)

                # Avoid error of picking a gem which is at the agent's position
                while (chosen_gem[0] == avatar_position[0] and
                       chosen_gem[1] == avatar_position[1]):
                    chosen_gem = gems[random.randint(0, len(gems) - 1)]

                # Search for a plan to chosen_gem
                self.action_list = self.search_plan(sso, chosen_gem)

                # Add sample to memory
                if self.EXECUTION_MODE == 'create_dataset' and self.is_training:
                    # Save current observation for dataset
                    one_hot_grid = self.encode_game_state(sso.observationGrid, chosen_gem)

                    # Save plan length as current metric
                    plan_metric = len(self.action_list)
                    
                    # <Append sample to dataset>

                    # First turn of the level -> only save s and r
                    if self.first_turn: 
                        self.first_turn = False
                        self.mem_sample = [one_hot_grid, plan_metric, None]

                    # Not the first turn -> add the current state s' to self.mem_sample, 
                    # create a new self.mem_sample and add to it s and r
                    else:
                        self.mem_sample[2] = sso
                        # Add old mem_sample to memory
                        self.memory.append(self.mem_sample)

                        # Create new mem_sample
                        self.mem_sample = [one_hot_grid, plan_metric, None]


        # Save dataset and exit the program if the experience replay is the right size

        if self.EXECUTION_MODE == 'create_dataset' and len(self.memory) >= self.num_samples_for_saving_dataset:
            self.save_dataset(self.dataset_save_path)

            # Exit the program with success code
            sys.exit()

        # Execute Plan

        # If a plan has been found, return the first action
        if len(self.action_list) > 0:
            if not self.is_training: # Count the number of actions used to complete the level
                self.num_actions_lv += 1

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

    def get_min_Q_value(self, sso):
        """
        Uses the Target Network (<with the current weights>) to obtain the Q-value associated with the state:
        the minimum Q-value among all possible gems present at that state.
        If sso is 'None' (corresponds to an end state), the Q-value is 0.

        @param sso Game state for which to calculate the Q-value.
        """

        # Check if sso is a terminal state (end of level)
        if sso is None:
            return 0

        observationGrid = sso.observationGrid
        min_Q_val = 1000000

        # Retrieve gems list (list of gems positions)
        gems = self.get_gems_positions(sso)

        for gem in gems:
            # Obtain one-hot encoding using the state (sso) and the current gem
            one_hot_grid = self.encode_game_state(observationGrid, gem)

            # Obtain the Q-value associated to that gem using the target network
            Q_val = self.target_network.predict(one_hot_grid)

            if Q_val < min_Q_val:
                min_Q_val = Q_val

        return min_Q_val

    def update_target_network(self):
        """
        Method used every tau steps to update the target network. It changes the target network's weights
        to those of the DQNetwork.
        """

        # Get the parameters of the DQNNetwork
        from_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")
        
        # Get the parameters of the Target_network
        to_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

        op_holder = []
        
        # Update our target_network parameters with DQNNetwork parameters
        for from_var,to_var in zip(from_vars,to_vars):
            op_holder.append(to_var.assign(from_var))

        return op_holder

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

    def save_dataset(self, path):
        """
        Uses the pickle module to save the experience replay to a file.

        @path Path of the file
        """

        print("\nSaving experience replay...")

        with open(path, 'wb') as file:
            pickle.dump(self.memory, file)

        print("Saving finished!")


    def load_dataset(self, path, num_elements=5000):
        """
        Uses the picle module to load the previously saved experience replay.

        @path Path of the file (without the '_<num_dataset>.dat' part)
        @num_elements The number of elements to load in total.
        """

        arr_indexes = [1,2,3,4,5,6,7,8,9,10]
        
        tam_each_dataset = 1000
        total_num_samples = 0

        print("\nLoading experience replay...")

        del self.memory[:] # Delete current array

        # Load datasets until num_elements elements are loaded
        while total_num_samples < num_elements:
            # Choose random dataset
            next_index = random.randint(0, len(arr_indexes)-1)
            next_dataset = arr_indexes[next_index]

            arr_indexes.remove(next_dataset) # Remove dataset from arr_indexes list (don't pick it again)

            print("Next dataset:", next_dataset)

            curr_path = path + "_{}.dat".format(next_dataset) # Next dataset to load

            with open(curr_path, 'rb') as file:
                curr_dataset = pickle.load(file)

                self.memory.extend(curr_dataset) # Append to memory

            print("{} loaded.".format(curr_path))

            total_num_samples += tam_each_dataset

        if len(self.memory) > num_elements:
            del self.memory[num_elements:] # Delete exceding elements

        print("Loading finished!\n")
        

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


