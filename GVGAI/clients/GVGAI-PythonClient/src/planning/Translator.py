import re
import yaml
import json

class Translator:
    def __init__(self, sso, config_path):
        """
        Translator constructor. Creates a new instance of a translator.

        @param sso State observation.
        @param config_path Path to the YAML configuration file.
        """
        # Save information about the map
        self.X_MAX = len(sso.observationGrid)
        self.Y_MAX = sso.observationGridMaxRow

        # Get correspondence between iType and name
        with open("logs/typeToName.json", "r") as f:
            json_content = json.load(f)
            self.type_to_name = {int(k): v for k, v in json_content.items()}

        # Load game information
        with open(config_path, "r") as stream:
            try:
                self.game_information = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        # Generate connectivity predicates and variables for each game element
        self.game_elements_vars = self._extract_variables_from_predicates()
        self.connection_predicates = self._generate_cell_conectivity_predicates()


    def translate_game_state_to_PDDL(self, sso, other_predicates):
        """
        Method that translates a game state observation to a PDDL state.

        @param sso State observation.
        @param other_predicates List of other predicates to be appended to the
                                resulting list of PDDL predicates that describe
                                the game's state.
        @return The state of the game as PDDL predicates and the objects that make
                it up.
        """
        pddl_predicates = []
        pddl_vars_instances = {var: set() for var in self.game_information["variablesTypes"].keys()}

        game_map = self._translate_state_observation(sso)

        for y in range(self.Y_MAX):
            for x in range(self.X_MAX):
                for element in game_map[y][x]:
                    if element in self.game_information["gameElementsCorrespondence"].keys():

                        for predicate in self.game_information["gameElementsCorrespondence"][element]:
                            predicate_instance = predicate

                            for var in self.game_elements_vars[element]:
                                if var in predicate_instance:
                                    if var == self.game_information["avatarVariable"]:
                                        var_instance = var.replace("?", "")

                                        if "orientationCorrespondence" in self.game_information.keys():
                                            if (sso.avatarOrientation[0] == 1.0):
                                                orientation = "RIGHT"
                                            elif (sso.avatarOrientation[0] == -1.0):
                                                orientation = "LEFT"
                                            elif (sso.avatarOrientation[1] == 1.0):
                                                orientation = "DOWN"
                                            elif (sso.avatarOrientation[1] == -1.0):
                                                orientation = "UP"

                                            pddl_predicates.append(self.game_information["orientationCorrespondence"][orientation].replace(var, var_instance))
                                    else:
                                        var_instance = f"{var}_{x}_{y}".replace("?", "")

                                    predicate_instance = predicate_instance.replace(var, var_instance)
                                    pddl_vars_instances[var].add(var_instance)

                            pddl_predicates.append(predicate_instance)

        # Add connections and other predicates
        pddl_predicates.extend(self.connection_predicates)
        pddl_predicates.extend(other_predicates)

        pddl_objects = {
                pddl_type: pddl_vars_instances[var]
                for var, pddl_type in self.game_information["variablesTypes"].items()
        }

        return pddl_predicates, pddl_objects

    
    def generate_goal_predicate(self, x, y):
        """
        Method used to generate a goal predicate. It is assumed that the goal
        predicate only contains an avatar variable and a cell variable.

        @param x X-axis coordinate that has to be reached.
        @param y Y-axis coordinate that has to be reached.
        """
        goal_predicate = self.game_information["goalPredicate"]

        avatar_instance = self.game_information["avatarVariable"].replace("?", "")
        cell_instance = f'{self.game_information["cellVariable"]}_{x}_{y}'.replace("?", "")

        goal_predicate = goal_predicate.replace(self.game_information["avatarVariable"], avatar_instance)
        goal_predicate = goal_predicate.replace(self.game_information["cellVariable"], cell_instance)

        return goal_predicate


    def translate_planner_output(self, planner_output):
        """
        Method that translates the planner's output as a list of GVGAI actions.

        @param planner_output Output generated by the planner.
        @return List of GVGAI actions to the current goal.
        """
        return [
                    self.game_information["actionsCorrespondence"][pddl_action]
                    for line in planner_output.split('\n')
                    for pddl_action in self.game_information["actionsCorrespondence"]
                    if pddl_action in line
        ]


    def _extract_variables_from_predicates(self):
        """
        Method used to extract the variables from the PDDL predicates and associate them
        to the corresponding game element.

        @return Correspondence between game elements and variables in their associated
                PDDL predicates.
        """
        return {
                obs: {
                    var
                    for predicate in predicates
                    for var in re.findall("\?[a-zA-Z]+", predicate)
                }
                for obs, predicates in self.game_information["gameElementsCorrespondence"].items()
        }


    def _generate_cell_conectivity_predicates(self):
        """
        Method used to generate the cell connectivity predicates.

        @return List containing the cell connectivity predicates.
        """
        def generate_connectivity_predicate(game_info, current_cell, connection, var, x, y):
            return game_info["connections"][connection].replace("?c", current_cell).replace(var, f'{self.game_information["cellVariable"]}_{x}_{y}'.replace("?", ""))

        game_info = self.game_information
        connection_predicates = []

        for y in range(self.Y_MAX):
            for x in range(self.X_MAX):
                current_cell = f'{self.game_information["cellVariable"]}_{x}_{y}'.replace("?", "")

                if y - 1 >= 0:
                    connection_predicates.append(generate_connectivity_predicate(game_info, current_cell, "UP", "?u", x, y - 1))

                if y + 1 < self.Y_MAX:
                    connection_predicates.append(generate_connectivity_predicate(game_info, current_cell, "DOWN", "?d", x, y + 1))

                if x - 1 >= 0:
                    connection_predicates.append(generate_connectivity_predicate(game_info, current_cell, "LEFT", "?l", x - 1, y))

                if x + 1 < self.X_MAX:
                    connection_predicates.append(generate_connectivity_predicate(game_info, current_cell, "RIGHT", "?r", x + 1, y))

        return connection_predicates


    def _translate_state_observation(self, sso):
        """
        Method that translates the state observation to the associated game
        elements in each position.

        @param sso State observation.
        @return List containing the game elements in each position.
        """
        return [
                    [
                        [
                            self.type_to_name[observation.itype] 
                            if observation is not None else "background"
                            for observation in sso.observationGrid[x][y]
                        ]
                        for x in range(self.X_MAX)
                    ]
                    for y in range(self.Y_MAX)
        ]

