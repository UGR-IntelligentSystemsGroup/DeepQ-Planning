from AbstractPlayer import AbstractPlayer
from Types import *

from planning.Translator import Translator

from utils.Types import LEARNING_SSO_TYPE
from planning.parser import Parser

import subprocess
import random
import pickle
import sys
import json
import yaml


class Agent(AbstractPlayer):

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
        self.translator = Translator(sso, "config/ice-and-fire.yaml")

        # Set first turn as True
        self.first_turn = True

        # Create new empty action list
        # This action list corresponds to the plan found by the planner
        self.action_list = []

        # It is true when the agent has the minimum required number of gems
        self.can_exit = False




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
        print(dir(sso))
        print(sso.avatarOrientation)
        #print(len(sso.observationGrid))
        #print(dir(sso.observationGrid[0][0]))
        #print(dir(sso.observationGrid[0][0][0]))
        #print(sso.observationGrid[0][0][0].category)
        #print(sso.observationGrid[0][0][0].itype)

        #print(sso.observationGrid[0][0][0].obsID)
        #print(sso.observationGrid[0][0][0].reference)
        pddl_predicates, pddl_objects = self.translator.translate_game_state_to_PDDL(sso, [])
        input()
        
        return 'ACTION_NIL'



