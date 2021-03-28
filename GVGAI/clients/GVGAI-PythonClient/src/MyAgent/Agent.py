from AbstractPlayer import AbstractPlayer
from Types import *

from planning.Translator import Translator
from planning.Planning import Planning

from utils.Types import LEARNING_SSO_TYPE
from planning.parser import Parser

import subprocess
import random
import numpy as np
import tensorflow as tf
import pickle
import sys
import glob
import time

# LearningModelFastRep.py is actually a little slower than LearningModel.py
from LearningModel import DQNetwork
from PrioritizedExperienceReplay import Memory

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

		# Attributes different for every game
		# Game in {'BoulderDash', 'IceAndFire', 'Catapults'}
		self.game_playing="Catapults"

		# Config file in {'config/boulderdash.yaml', 'config/ice-and-fire.yaml', 'config/catapults.yaml'}
		if self.game_playing == 'BoulderDash':
			self.config_file='config/boulderdash.yaml'
		elif self.game_playing == 'IceAndFire':
			self.config_file='config/ice-and-fire.yaml'
		else: # Catapults
			self.config_file='config/catapults.yaml'
		self.planning = Planning(self.config_file)

		# The number of actions an invalid plan is associated, i.e., when
		# there is no plan for the selected subgoal
		self.num_actions_invalid_plan = 1000

		# << Choose execution mode >>
		# - 'create_dataset' -> It doesn't train any model. Just creates the dataset (experience replay) and saves it
		# - 'train' -> It loads the saved dataset, trains the model with it, and saves the trained model. 
		#              It doesn't add any sample to the experience replay
		# - 'test' -> It loads the trained model and tests it on the validation levels, obtaining the metrics.


		self.EXECUTION_MODE="train"

		# Size of the dataset to train the model on
		self.dataset_size_for_training=200

		# Name of the DQNetwork. Also used for creating the name of file to save and load the model from
		# Add the name of the game being played!!!
		self.network_name="DQN_Catapults_NO_persistent_tensors_20_rep_gamma-1_its-1000000_Catapults_1"
		self.network_name=self.network_name + "_lvs={}".format(self.dataset_size_for_training)

		# Seed for selecting which levels to train the model on
		self.level_seed=57824

		# <Model Hyperparameters>
		# Automatically changed by ejecutar_pruebas.py!

		# Architecture
		# First conv layer
		self.l1_num_filt=32
		self.l1_window=[5, 5]
		self.l1_strides=[1, 1]
		self.l1_padding_type="VALID"

		# Second conv layer
		self.l2_num_filt=64
		self.l2_window=[5, 5]
		self.l2_strides=[1, 1]
		self.l2_padding_type="VALID"

		# Third conv layer
		self.l3_num_filt=64
		self.l3_window=[5, 5]
		self.l3_strides=[1, 1]
		self.l3_padding_type="VALID"

		# Fourth conv layer
		self.l4_num_filt=-1
		self.l4_window=[3, 3]
		self.l4_strides=[1, 1]
		self.l4_padding_type="VALID"

		# Fifth conv layer
		self.l5_num_filt=-1
		self.l5_window=[3, 3]
		self.l5_strides=[1, 1]
		self.l5_padding_type="VALID"

		# Sixth conv layer
		self.l6_num_filt=-1
		self.l6_window=[3, 3]
		self.l6_strides=[1, 1]
		self.l6_padding_type="VALID"

		self.l7_num_filt=-1
		self.l7_window=[3, 3]
		self.l7_strides=[1, 1]
		self.l7_padding_type="VALID"

		self.l8_num_filt=-1
		self.l8_window=[3, 3]
		self.l8_strides=[1, 1]
		self.l8_padding_type="VALID"

		self.l9_num_filt=-1
		self.l9_window=[3, 3]
		self.l9_strides=[1, 1]
		self.l9_padding_type="VALID"

		self.l10_num_filt=-1
		self.l10_window=[3, 3]
		self.l10_strides=[1, 1]
		self.l10_padding_type="VALID"

		self.l11_num_filt=-1
		self.l11_window=[3, 3]
		self.l11_strides=[1, 1]
		self.l11_padding_type="SAME"

		self.l12_num_filt=-1
		self.l12_window=[3, 3]
		self.l12_strides=[1, 1]
		self.l12_padding_type="VALID"

		self.l13_num_filt=-1
		self.l13_window=[3, 3]
		self.l13_strides=[1, 1]
		self.l13_padding_type="SAME"

		self.l14_num_filt=-1
		self.l14_window=[3, 3]
		self.l14_strides=[1, 1]
		self.l14_padding_type="VALID"

		self.l15_num_filt=-1
		self.l15_window=[3, 3]
		self.l15_strides=[1, 1]
		self.l15_padding_type="VALID"

		self.l16_num_filt=-1
		self.l16_window=[3, 3]
		self.l16_strides=[1, 1]
		self.l16_padding_type="VALID"

		self.l17_num_filt=-1
		self.l17_window=[3, 3]
		self.l17_strides=[1, 1]
		self.l17_padding_type="SAME"

		self.l18_num_filt=-1
		self.l18_window=[3, 3]
		self.l18_strides=[1, 1]
		self.l18_padding_type="VALID"

		self.l19_num_filt=-1
		self.l19_window=[3, 3]
		self.l19_strides=[1, 1]
		self.l19_padding_type="VALID"

		self.l20_num_filt=-1
		self.l20_window=[3, 3]
		self.l20_strides=[1, 1]
		self.l20_padding_type="VALID"

		# Number of units of the first and second fully-connected layers
		self.fc_num_unis=[128, 1, 1, 1]

		# Training params
		self.learning_rate=0.0001
		# Don't use dropout?
		self.dropout_prob=0.0
		self.num_train_its=1000000
		self.batch_size=32
		self.use_BN=False
		
		# Extra params
		# Number of training its before copying the DQNetwork's weights to the target network
		# default max_tau was 250
		self.max_tau=1000
		self.tau=0 # Counter that resets to 0 when the target network is updated
		# Discount rate for Deep Q-Learning
		self.gamma=1

		# Sample size. It depens on the game being played. The format is (rows, cols, number of observations + 1)
		# Sizes: BoulderDash=[13, 26, 9], IceAndFire=[14, 16, 10] , Catapults=[16, 16, 9]
		if self.game_playing == 'BoulderDash':
			self.sample_size=[13, 26, 7]
		elif self.game_playing == 'IceAndFire':
			self.sample_size=[14, 16, 10]
		else: # Catapults
			self.sample_size=[16, 16, 9]

		if self.EXECUTION_MODE == 'create_dataset':

			# Attribute that stores agent's experience to implement experience replay (for training)
			# Each element is a tuple (s, res, r, s') corresponding to:
			# - s -> start_state and chosen subgoal ((game state, chosen subgoal) one-hot encoded). Shape = (-1, 13, 26, 9).
			# - res -> list of three elements containing the resources the agent has at the state s.
			# - r -> length of the plan from the start_state to the chosen subgoal.
			# - s' -> state of the game just after the agent has achieved the chosen subgoal.
			#         It's an instance of the SerializableStateObservation (sso) class.
			self.memory = []

			self.total_num_samples = 0 # Total number of sample collected, even if they were not unique (thus not added to the datasets)
			self.sample_hashes = set() # Hashes of unique samples already collected

			# Path of the file to save the experience replay to
			id_dataset=82
			self.dataset_save_path = 'SavedDatasets/' + 'dataset_{}_{}.dat'.format(self.game_playing, id_dataset)
			# Path of the file which contains the number of samples of each saved dataset
			self.datasets_sizes_file_path = 'SavedDatasets/Datasets Sizes.txt'

			# Size of the experience replay to save. It is saved when the total number of samples collected reaches
			# 'self.num_total_samples_for_saving_dataset' or when the unique number of samples (len(self.memory)) reaches
			# 'self.num_unique_samples_for_saving_dataset'
			self.num_total_samples_for_saving_dataset = 1000
			self.num_unique_samples_for_saving_dataset = 500

		elif self.EXECUTION_MODE == 'train':
			# Name of the saved model file (without the number of dataset size part)
			self.model_save_path = "./SavedModels/" + self.network_name + ".ckpt"

			# Array with samples
			self.memory = []

			# Prioritized Experience Replay
			self.PER = None

			# Folder where the datasets are stored
			self.datasets_folder = 'SavedDatasets'

			# Period for saving the trained model -> the model is saved every X training iterations
			self.num_its_each_model_save = 100000

			# If true, PER is used. Otherwise, random sampling is used.
			self.use_PER=True

			# Each time self.model.train is called, this variable controls how many train
			# repetitions are performed
			self.num_repetitions_each_train_call = 20

		else: # Test

			# Goal Selection Mode: "best" -> select the best one using the trained model,
			# "random" -> select a random one (corresponds with the random model)
			# "greedy" -> plans to every subgoal and selects the one with the shortest plan
			# Automatically changed by the scripts!
			self.goal_selection_mode="best"

			# Create Learning Model unles goal selection model is random

			if self.goal_selection_mode == "best":
				# DQNetwork
				self.model = DQNetwork(writer_name=self.network_name,
						 sample_size = self.sample_size,
						 l1_num_filt = self.l1_num_filt, l1_window = self.l1_window, l1_strides = self.l1_strides,
						 l1_padding_type = self.l1_padding_type,
						 l2_num_filt = self.l2_num_filt, l2_window = self.l2_window, l2_strides = self.l2_strides,
						 l2_padding_type = self.l2_padding_type,
						 l3_num_filt = self.l3_num_filt, l3_window = self.l3_window, l3_strides = self.l3_strides,
						 l3_padding_type = self.l3_padding_type,
						 l4_num_filt = self.l4_num_filt, l4_window = self.l4_window, l4_strides = self.l4_strides,
						 l4_padding_type = self.l4_padding_type,
						 l5_num_filt = self.l5_num_filt, l5_window = self.l5_window, l5_strides = self.l5_strides,
						 l5_padding_type = self.l5_padding_type,
						 l6_num_filt = self.l6_num_filt, l6_window = self.l6_window, l6_strides = self.l6_strides,
						 l6_padding_type = self.l6_padding_type,
						 l7_num_filt = self.l7_num_filt, l7_window = self.l7_window, l7_strides = self.l7_strides,
						 l7_padding_type = self.l7_padding_type,
						 l8_num_filt = self.l8_num_filt, l8_window = self.l8_window, l8_strides = self.l8_strides,
						 l8_padding_type = self.l8_padding_type,
						 l9_num_filt = self.l9_num_filt, l9_window = self.l9_window, l9_strides = self.l9_strides,
						 l9_padding_type = self.l9_padding_type,
						 l10_num_filt = self.l10_num_filt, l10_window = self.l10_window, l10_strides = self.l10_strides,
						 l10_padding_type = self.l10_padding_type,
						 l11_num_filt = self.l11_num_filt, l11_window = self.l11_window, l11_strides = self.l11_strides,
						 l11_padding_type = self.l11_padding_type,
						 l12_num_filt = self.l12_num_filt, l12_window = self.l12_window, l12_strides = self.l12_strides,
						 l12_padding_type = self.l12_padding_type,
						 l13_num_filt = self.l13_num_filt, l13_window = self.l13_window, l13_strides = self.l13_strides,
						 l13_padding_type = self.l13_padding_type,
						 l14_num_filt = self.l14_num_filt, l14_window = self.l14_window, l14_strides = self.l14_strides,
						 l14_padding_type = self.l14_padding_type,
						 l15_num_filt = self.l15_num_filt, l15_window = self.l15_window, l15_strides = self.l15_strides,
						 l15_padding_type = self.l15_padding_type,
						 l16_num_filt = self.l16_num_filt, l16_window = self.l16_window, l16_strides = self.l16_strides,
						 l16_padding_type = self.l16_padding_type,
						 l17_num_filt = self.l17_num_filt, l17_window = self.l17_window, l17_strides = self.l17_strides,
						 l17_padding_type = self.l17_padding_type,
						 l18_num_filt = self.l18_num_filt, l18_window = self.l18_window, l18_strides = self.l18_strides,
						 l18_padding_type = self.l18_padding_type,
						 l19_num_filt = self.l19_num_filt, l19_window = self.l19_window, l19_strides = self.l19_strides,
						 l19_padding_type = self.l19_padding_type,
						 l20_num_filt = self.l20_num_filt, l20_window = self.l20_window, l20_strides = self.l20_strides,
						 l20_padding_type = self.l20_padding_type,
						 fc_num_units = self.fc_num_unis, dropout_prob = 0.0,
						 learning_rate = self.learning_rate,
						 use_BN=self.use_BN, game_playing=self.game_playing)

				# Name of the saved model file to load (without the number of training steps part)
				model_load_path = "./SavedModels/" + self.network_name + ".ckpt"

				# Number training its of the model to load
				# Automatically changed by ejecutar_pruebas.py!
				self.num_train_its_model=1000000

				# <Load the already-trained model in order to test performance>
				self.model.load_model(path = model_load_path, num_it = self.num_train_its_model)

			# Number of test levels the agent is playing. If it's 1, the agent exits after playing only the first test level
			# Automatically changed by ejecutar_pruebas.py!
			self.num_test_levels=1

			# If True, the agent has already finished the first test level and is playing the second one
			self.playing_second_test_level = False


	def init(self, sso, elapsedTimer):
		"""
		* Public method to be called at the start of every level of a game.
		* Perform any level-entry initialization here.
		* @param sso Phase Observation of the current game.
		* @param elapsedTimer Timer, which is 1s by default. Modified to 1000s.
							  Check utils/CompetitionParameters.py for more info.
		"""
		self.translator = Translator(sso, self.config_file)

		# Create empty mem_sample
		if self.EXECUTION_MODE == 'create_dataset':
			self.mem_sample = []

		# Create new empty action list
		# This action list corresponds to the plan found by the planner
		self.action_list = []

		# See if it's training or validation time
		self.is_training = not sso.isValidation # It's the opposite to sso.isValidation

		# If it's validation/test phase, count the number of actions used
		# to beat the current level and the number of incorrect subgoals
		# (not eligible) the agent selects
		if self.EXECUTION_MODE == 'test' and not self.is_training:
			self.num_actions_lv = 0
			self.num_incorrect_subgoals = 0

			# Measure the goal selection + planning times
			self.total_time_planning_curr_lv = 0
			self.total_time_goal_selec_curr_lv = 0


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
			# Load dataset of current size
			self.load_dataset(self.datasets_folder, self.game_playing, num_levels=self.dataset_size_for_training, seed=self.level_seed)

			# Shuffle dataset
			# DO NOT USE random.shuffle (it does not work well with numpy arrays)
			np.random.shuffle(self.memory)

			# Create Prioritized Experience Replay
			if self.use_PER:
				self.PER = Memory(len(self.memory), self.memory)

			# Create Learning model

			tf.reset_default_graph() # Clear Tensorflow Graph and Variables

			# DQNetwork
			self.model = DQNetwork(writer_name=self.network_name,
					 sample_size = self.sample_size,
					 l1_num_filt = self.l1_num_filt, l1_window = self.l1_window, l1_strides = self.l1_strides,
					 l1_padding_type = self.l1_padding_type,
					 l2_num_filt = self.l2_num_filt, l2_window = self.l2_window, l2_strides = self.l2_strides,
					 l2_padding_type = self.l2_padding_type,
					 l3_num_filt = self.l3_num_filt, l3_window = self.l3_window, l3_strides = self.l3_strides,
					 l3_padding_type = self.l3_padding_type,
					 l4_num_filt = self.l4_num_filt, l4_window = self.l4_window, l4_strides = self.l4_strides,
					 l4_padding_type = self.l4_padding_type,
					 l5_num_filt = self.l5_num_filt, l5_window = self.l5_window, l5_strides = self.l5_strides,
					 l5_padding_type = self.l5_padding_type,
					 l6_num_filt = self.l6_num_filt, l6_window = self.l6_window, l6_strides = self.l6_strides,
					 l6_padding_type = self.l6_padding_type,
					 l7_num_filt = self.l7_num_filt, l7_window = self.l7_window, l7_strides = self.l7_strides,
					 l7_padding_type = self.l7_padding_type,
					 l8_num_filt = self.l8_num_filt, l8_window = self.l8_window, l8_strides = self.l8_strides,
					 l8_padding_type = self.l8_padding_type,
					 l9_num_filt = self.l9_num_filt, l9_window = self.l9_window, l9_strides = self.l9_strides,
					 l9_padding_type = self.l9_padding_type,
					 l10_num_filt = self.l10_num_filt, l10_window = self.l10_window, l10_strides = self.l10_strides,
					 l10_padding_type = self.l10_padding_type,
					 l11_num_filt = self.l11_num_filt, l11_window = self.l11_window, l11_strides = self.l11_strides,
					 l11_padding_type = self.l11_padding_type,
					 l12_num_filt = self.l12_num_filt, l12_window = self.l12_window, l12_strides = self.l12_strides,
					 l12_padding_type = self.l12_padding_type,
					 l13_num_filt = self.l13_num_filt, l13_window = self.l13_window, l13_strides = self.l13_strides,
					 l13_padding_type = self.l13_padding_type,
					 l14_num_filt = self.l14_num_filt, l14_window = self.l14_window, l14_strides = self.l14_strides,
					 l14_padding_type = self.l14_padding_type,
					 l15_num_filt = self.l15_num_filt, l15_window = self.l15_window, l15_strides = self.l15_strides,
					 l15_padding_type = self.l15_padding_type,
					 l16_num_filt = self.l16_num_filt, l16_window = self.l16_window, l16_strides = self.l16_strides,
					 l16_padding_type = self.l16_padding_type,
					 l17_num_filt = self.l17_num_filt, l17_window = self.l17_window, l17_strides = self.l17_strides,
					 l17_padding_type = self.l17_padding_type,
					 l18_num_filt = self.l18_num_filt, l18_window = self.l18_window, l18_strides = self.l18_strides,
					 l18_padding_type = self.l18_padding_type,
					 l19_num_filt = self.l19_num_filt, l19_window = self.l19_window, l19_strides = self.l19_strides,
					 l19_padding_type = self.l19_padding_type,
					 l20_num_filt = self.l20_num_filt, l20_window = self.l20_window, l20_strides = self.l20_strides,
					 l20_padding_type = self.l20_padding_type,
					 fc_num_units = self.fc_num_unis, dropout_prob = self.dropout_prob,
					 learning_rate = self.learning_rate,
					 use_BN=self.use_BN, game_playing=self.game_playing)

			# Target Network
			# Used to predict the Q targets. It is upgraded every max_tau updates.
			# Use the same tf.session as the main DQNetwork
			self.target_network = DQNetwork(name="TargetNetwork",
					 sess = self.model.sess,
					 create_writer = False,
					 sample_size = self.sample_size,
					 l1_num_filt = self.l1_num_filt, l1_window = self.l1_window, l1_strides = self.l1_strides,
					 l1_padding_type = self.l1_padding_type,
					 l2_num_filt = self.l2_num_filt, l2_window = self.l2_window, l2_strides = self.l2_strides,
					 l2_padding_type = self.l2_padding_type,
					 l3_num_filt = self.l3_num_filt, l3_window = self.l3_window, l3_strides = self.l3_strides,
					 l3_padding_type = self.l3_padding_type,
					 l4_num_filt = self.l4_num_filt, l4_window = self.l4_window, l4_strides = self.l4_strides,
					 l4_padding_type = self.l4_padding_type,
					 l5_num_filt = self.l5_num_filt, l5_window = self.l5_window, l5_strides = self.l5_strides,
					 l5_padding_type = self.l5_padding_type,
					 l6_num_filt = self.l6_num_filt, l6_window = self.l6_window, l6_strides = self.l6_strides,
					 l6_padding_type = self.l6_padding_type,
					 l7_num_filt = self.l7_num_filt, l7_window = self.l7_window, l7_strides = self.l7_strides,
					 l7_padding_type = self.l7_padding_type,
					 l8_num_filt = self.l8_num_filt, l8_window = self.l8_window, l8_strides = self.l8_strides,
					 l8_padding_type = self.l8_padding_type,
					 l9_num_filt = self.l9_num_filt, l9_window = self.l9_window, l9_strides = self.l9_strides,
					 l9_padding_type = self.l9_padding_type,
					 l10_num_filt = self.l10_num_filt, l10_window = self.l10_window, l10_strides = self.l10_strides,
					 l10_padding_type = self.l10_padding_type,
					 l11_num_filt = self.l11_num_filt, l11_window = self.l11_window, l11_strides = self.l11_strides,
					 l11_padding_type = self.l11_padding_type,
					 l12_num_filt = self.l12_num_filt, l12_window = self.l12_window, l12_strides = self.l12_strides,
					 l12_padding_type = self.l12_padding_type,
					 l13_num_filt = self.l13_num_filt, l13_window = self.l13_window, l13_strides = self.l13_strides,
					 l13_padding_type = self.l13_padding_type,
					 l14_num_filt = self.l14_num_filt, l14_window = self.l14_window, l14_strides = self.l14_strides,
					 l14_padding_type = self.l14_padding_type,
					 l15_num_filt = self.l15_num_filt, l15_window = self.l15_window, l15_strides = self.l15_strides,
					 l15_padding_type = self.l15_padding_type,
					 l16_num_filt = self.l16_num_filt, l16_window = self.l16_window, l16_strides = self.l16_strides,
					 l16_padding_type = self.l16_padding_type,
					 l17_num_filt = self.l17_num_filt, l17_window = self.l17_window, l17_strides = self.l17_strides,
					 l17_padding_type = self.l17_padding_type,
					 l18_num_filt = self.l18_num_filt, l18_window = self.l18_window, l18_strides = self.l18_strides,
					 l18_padding_type = self.l18_padding_type,
					 l19_num_filt = self.l19_num_filt, l19_window = self.l19_window, l19_strides = self.l19_strides,
					 l19_padding_type = self.l19_padding_type,
					 l20_num_filt = self.l20_num_filt, l20_window = self.l20_window, l20_strides = self.l20_strides,
					 l20_padding_type = self.l20_padding_type,
					 fc_num_units = self.fc_num_unis, dropout_prob = 0.0,
					 learning_rate = self.learning_rate,
					 use_BN=self.use_BN, game_playing=self.game_playing)

			# Initialize target network's weights with those of the DQNetwork
			self.update_ops = self.update_target_network() # ONLY CALL THIS ONCE (else new nodes will be added to the graph with each iter)
			self.target_network.update_weights(self.update_ops)
			self.tau = 0

			num_samples = len(self.memory)

			print("\n> Started training of model on {} levels\n".format(self.dataset_size_for_training))

			ind_batch = 0 # Index for selecting the next minibatch

			# Execute the training of the current model
			curr_it = 0

			while curr_it < self.num_train_its:   
				# Choose next batch from Experience Replay (using PER or random sampling)

				# PER
				if self.use_PER:
					tree_idx, batch, sample_weights = self.PER.sample(self.batch_size)	
				else: # Random Sampling
					if ind_batch+self.batch_size < num_samples:
						batch = self.memory[ind_batch:ind_batch+self.batch_size]
						ind_batch = (ind_batch + self.batch_size)
					else: # Got to the end of the experience replay -> shuffle it and start again
						batch = self.memory[ind_batch:]
						ind_batch = 0

						np.random.shuffle(self.memory)


				batch_X = np.array([each[0] for each in batch]) # inputs for the DQNetwork
				# batch_Res = np.array([each[1] for each in batch]) # List of Agent Resources for the DQNetwork
				batch_R = [each[2] for each in batch] # r values (plan lenghts)
				batch_S = [each[3] for each in batch] # s' values (sso instances)

				# Calculate Q_targets
				Q_targets = []
				
				for r, s in zip(batch_R, batch_S):
					# Modify the reward when the current state is terminal (the next
					# state s is None)
					if s is None:
						# Invalid plan -> penalization
						if r >= 1000:
							r = 100
						else: # Valid plan -> reward
							r = -100

					Q_target = r + self.gamma*self.get_min_Q_value(s)

					Q_targets.append(Q_target)

				Q_targets = np.reshape(Q_targets, (-1, 1)) 

				# Execute one training step
				# absolute_errors is used to update the priority scores of the PER	
				if self.use_PER:			
					absolute_errors = self.model.train(batch_X, Q_targets, sample_weights, num_its = self.num_repetitions_each_train_call)
				else: # If we are not using PER, don't pass sample_weights
					self.model.train(batch_X, Q_targets, num_its = self.num_repetitions_each_train_call)

				self.tau += 1

				# Update the priority scores of the PER
				if self.use_PER:
					self.PER.batch_update(tree_idx, absolute_errors)

				# Update target network every tau training steps
				if self.tau >= self.max_tau:
					# update_ops = self.update_target_network()
					self.target_network.update_weights(self.update_ops)

					self.tau = 0

				# Save Logs every 1000 training its
				if curr_it % 1000 == 0:
					self.model.save_logs(batch_X, Q_targets, curr_it)

				# Save the model each X training its
				if curr_it > 0 and curr_it % self.num_its_each_model_save == 0:
					self.model.save_model(path = self.model_save_path, num_it = curr_it)


				# Periodically print the progress of the training
				if curr_it % 500 == 0 and curr_it != 0:
					print("- {} its completed".format(curr_it))

				# Update the curr_it taking into account how many train its are performed
				# in each self.model.train call
				curr_it += self.num_repetitions_each_train_call
																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																							
			# Save the current trained model
			self.model.save_model(path = self.model_save_path, num_it = self.num_train_its)
			print("\n> Current model saved! Dataset size={} levels\n".format(self.dataset_size_for_training))


			# Exit the program after finishing training
			print("\nTraining finished!")
			sys.exit()
  

		# If the agent is in test mode but is currently at a training level, it escapes the level
		if self.EXECUTION_MODE=="test" and self.is_training:
			return 'ACTION_ESCAPE'


		# <Play the game (EXECUTION_MODE == 'create_dataset' or 'test')>

		# Check if the agent can act at the current game state, i.e., execute an action.
		# If it can't, the agent returns 'ACTION_NIL'
		if not self.can_act(sso):
			return 'ACTION_NIL'

		# If the plan is empty, get a new one
		if len(self.action_list) == 0:
			# Get eligible subgoals at the current state
			subgoals = self.get_subgoals_positions(sso)

			# Make sure no subgoal is at the agent's position
			avatar_position = (int(sso.avatarPosition[0] // sso.blockSize),
								 int(sso.avatarPosition[1] // sso.blockSize))
			
			if avatar_position in subgoals:
				subgoals.remove(avatar_position)

			# Choose a random subgoal if the agent is randomly exploring
			if self.EXECUTION_MODE=="create_dataset":
				# Choose a random subgoal until one is attainable (search_plan returns a non-empty plan)
				while len(subgoals) > 0 and len(self.action_list) == 0: 
					ind_subgoal = random.randint(0, len(subgoals) - 1)

					chosen_subgoal = subgoals[ind_subgoal]
					del subgoals[ind_subgoal] # Delete the chosen subgoal from the list

					# If the game is IceAndFire, check how many types of boots the agent has
					if self.game_playing == 'IceAndFire': 
						boots_resources = self.get_boots_resources(sso)
					else:
						boots_resources = []

					# Obtain the plan
					self.action_list = self.search_plan(sso, chosen_subgoal, boots_resources)
					
					# Collect samples for the experience replay
					self.add_samples_to_memory(sso, chosen_subgoal, len(self.action_list))
					self.total_num_samples += 1

					# Show the number of samples collected periodically
					print("New sample")

					if self.total_num_samples % 10 == 0:
						print("n_total_samples:", self.total_num_samples)
						print("n_unique_samples:", len(self.memory))

				# If there is no plan, it means that at the current game state no subgoal is attainable ->
				# the agent escapes the level (loses the game)
				if len(self.action_list) == 0:
					return 'ACTION_ESCAPE'

			# Use the Learning model to select a subgoal if the agent is in test/validation phase
			else:
				ordered_subgoals = False # If true, get_best_subgoals has already been called

				# Keep selecting subgoal until one is attainable from the current state of the game
				while len(subgoals) > 0 and len(self.action_list) == 0:

					# Select the subgoal using either the learning model or randomly (random model)
					if self.goal_selection_mode == "best": # Use the model to select the best subgoal

						# Order subgoals by their predicted Q_values
						if not ordered_subgoals: # Only do it once
							ordered_subgoals = True

							# Measure goal selection time
							start = time.time()

							subgoals = self.get_best_subgoals(sso, subgoals)

							end = time.time()

							if not self.is_training:
								self.total_time_goal_selec_curr_lv += end-start

						# Get the first subgoal (the one with the smallest Q_value)
						chosen_subgoal = subgoals[0]

					elif self.goal_selection_mode == "random": # Select subgoals randomly
						chosen_subgoal = subgoals[random.randint(0, len(subgoals) - 1)]

					# goal selection mode = greedy
					else: # Plan to every subgoal and select the one of the shortest plan

						# Order subgoals by their plan lengths
						if not ordered_subgoals: # Only do it once
							ordered_subgoals = True

							# Measure planning times
							start = time.time()

							old_subgoals = subgoals
							subgoals = self.order_subgoals_by_plan_length(sso, subgoals)

							end = time.time()

							if not self.is_training:
								self.total_time_planning_curr_lv += end-start

						# Get the first subgoal (the one with the smallest plan length)
						if len(subgoals) > 0:
							chosen_subgoal = subgoals[0]
						else: # If there is no valid subgoal, choose any subgoal
							chosen_subgoal = old_subgoals[0]

					# Remove the selected subgoal from the list of eligible subgoals
					if len(subgoals) > 0:
						subgoals.remove(chosen_subgoal)

					# If the game is IceAndFire, check how many types of boots the agent has
					if self.game_playing == 'IceAndFire': 
						boots_resources = self.get_boots_resources(sso)
					else:
						boots_resources = []

					# Obtain the plan

					start = time.time()

					self.action_list = self.search_plan(sso, chosen_subgoal, boots_resources)

					end = time.time()

					# Measure planning time unless the goal selection mode is greedy, because in that case
					# we have already planned for each subgoal
					# (we could actually obtain a list of plans for each possible subgoal in that case but
					# to simplify the code we plan for each subgoal again (even though it's not needed) and
					# don't add the planning time)
					if not self.is_training and self.goal_selection_mode != "greedy":
						self.total_time_planning_curr_lv += end-start

					# If there is no valid plan to the chosen subgoal, increment the number of
					# non-eligible subgoals selected
					if len(self.action_list) == 0:
						self.num_incorrect_subgoals += 1 

				# If none of the subgoals is attainable, the agent is at a dead end ->
				# the agent loses the game and escapes the level
				if len(self.action_list) == 0:
					self.num_incorrect_subgoals = -1 # This represents the agent has lost the game
					return 'ACTION_ESCAPE'

		# Save dataset and exit the program if the experience replay is the right size
		if self.EXECUTION_MODE == 'create_dataset':
			if self.total_num_samples >= self.num_total_samples_for_saving_dataset or \
			   len(self.memory) >= self.num_unique_samples_for_saving_dataset:
			
				self.save_dataset(self.dataset_save_path, self.datasets_sizes_file_path)

				# Exit the program with success code
				sys.exit()

		# Execute Plan

		# If a plan has been found, return the first action
		if len(self.action_list) > 0:
			if not self.is_training: # Count the number of actions used to complete the level
				self.num_actions_lv += 1

			return self.action_list.pop(0)
		else:
			print("\n\nEMPTY PLAN\n\n")
			return 'ACTION_NIL'


	def add_samples_to_memory(self, sso, chosen_subgoal, plan_length):
		"""
		Method called at the 'create_dataset' phase, when a new plan (empty or not)
		has been obtained for a given (sso, chosen_subgoal) pair.
		Firstly, it completes the old mem_sample (if needed) and adds it to the experience replay.
		Secondly, it creates a new mem_sample and adds the one_hot_matrix and plan_length to it.
		If the chosen_subgoal corresponds to the final goal or the plan is invalid,
		the new mem_sample is already added to the experience replay. If not, it will be added
		the next time this method is called.
		
		@param sso The current state of the game
		@param chosen_subgoal The selected subgoal for the current state of the game
		@param plan_length The number of actions of the plan obtained for the (sso, chosen_subgoal) pair.
						   If it's 0, the plan is invalid.
		"""
		# Obtain the resources of the agent at the current state of the game given by sso
		agent_resources = self.get_agent_resources(sso)

		# If the old mem_sample lacks the new state of the game, complete it
		# and add it to the experience replay
		if len(self.mem_sample) == 3:
			self.mem_sample.append(sso) # Add the end state (the current state of the game)
			self.memory.append(self.mem_sample) # Add old mem_sample to memory

		# <Check if the new sample is unique>

		# Obtain the hash for the (sso, chosen_subgoal) pair
		new_hash = self.get_sso_subgoal_hash(sso, chosen_subgoal)

		# If the hash is new, create a new sample
		if new_hash not in self.sample_hashes:
			# Add the hash
			self.sample_hashes.add(new_hash) 

			# Encode (sso, chosen_subgoal) as a one_hot matrix
			one_hot_grid = self.encode_game_state(sso.observationGrid, chosen_subgoal)

			# Invalid plan (there is no plan for the chosen_subgoal)
			if plan_length == 0:
				# If the plan is invalid, its length will be saved as self.num_actions_invalid_plan
				plan_length = self.num_actions_invalid_plan  

				# Save it already to the experience replay (since there is no end state)
				self.mem_sample = [one_hot_grid, agent_resources, plan_length, None]

				self.memory.append(self.mem_sample)
			
			# Valid plan
			else:
				# Check if the subgoal corresponds to the final goal (exit)
				exit = sso.portalsPositions[0][0]
				exit_pos_x = int(exit.position.x // sso.blockSize)
				exit_pos_y = int(exit.position.y // sso.blockSize)

				subgoal_is_final = (exit_pos_x == chosen_subgoal[0] and
								 exit_pos_y == chosen_subgoal[1])

				if subgoal_is_final:
					# Save the new sample already to the experience replay (since there is no end state)
					self.mem_sample = [one_hot_grid, agent_resources, plan_length, None]
					
					self.memory.append(self.mem_sample)
				else:
					# Create the new, incomplete mem_sample
					self.mem_sample = [one_hot_grid, agent_resources, plan_length]


	def get_agent_resources(self, sso):
		"""
		Returns a list with three elements, that contains the resources the agent has in
		the state given by sso. These resources depend on the game being played.
		"""

		# BoulderDash -> [gems]
		if self.game_playing == 'BoulderDash':
			keys = sso.avatarResources.keys()

			if len(keys) > 0:
				gem_key = list(sso.avatarResources)[0]
				num_gems = sso.avatarResources[gem_key]
			else:
				num_gems = 0

			agent_resources = [num_gems, 0, 0]

		# IceAndFire -> [coins, fire_boots, ice_boots]
		elif self.game_playing == "IceAndFire":
			# Get the number of coins left on the map
			coin_itype = 10 # Itype of coins

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

			# Each level always has 10 coins at the start
			num_coins_agent = 10 - coins_on_the_map

			# Get the boots the agent has
			keys = sso.avatarResources.keys()
			ice_boot_key = '8'  # Keys of each boot in the avatarResources dictionary
			fire_boot_key = '9'                     

			ice_boots_agent = 0
			fire_boots_agent = 0
			# Check if the agent has at least one boot  
			if len(keys) > 0:
				if ice_boot_key in keys: # The agent has ice boots
					ice_boots_agent = 1
				if fire_boot_key in keys: # The agent has fire boots
					fire_boots_agent = 1
			
			agent_resources = [num_coins_agent, fire_boots_agent, ice_boots_agent]

		# Catapults -> no resources
		else:
			agent_resources = [0,0,0]

		return agent_resources
	

	def get_subgoals_positions(self, sso):
		"""
		Method that returns all the eligible subgoals by the agent at the current state of 
		the game.
		Note: the final subgoal is always returned, even if it's not attainable.
		
		@param sso Observation of the current state of the game.
		@return The grid positions of the eligible subgoals, as a list of the (x,y) coordinates
				of each subgoal.
		"""

		if self.game_playing == 'BoulderDash':
			subgoal_pos = [] # Positions of subgoals

			# Final goal
			exit = sso.portalsPositions[0][0]
			exit_pos = (int(exit.position.x // sso.blockSize), int(exit.position.y // sso.blockSize))
			subgoal_pos.append(exit_pos)
			
			# Gems subgoals

			# Check if there is at least one gem in the level
			if len(sso.resourcesPositions) > 0:
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

	def get_boots_resources(self, sso):
		"""
		Method used only if self.game_playing=='IceAndFire'. It returns a list with
		the types of boots (ice, fire, none or both) the agent currently has.
		
		@param sso Observation of the current state of the game.
		@return A list containing the types of boots the agent has. This list
				will be passed as the 'other_predicates' parameters to self.search_plan().
		"""

		boots_resources = []

		keys = sso.avatarResources.keys()
		ice_boot_key = '8'  # Keys of each boot in the avatarResources dictionary
		fire_boot_key = '9'                     

		# Check if the agent has at least one boot  
		if len(keys) > 0:
			if ice_boot_key in keys: # The agent has ice boots
				boots_resources.append("(has-ice-boots)")
			if fire_boot_key in keys: # The agent has fire boots
				boots_resources.append("(has-fire-boots)")

		return boots_resources


	def can_act(self, sso):
		"""
		This method returns True if the agent can act, i.e., can execute an action different than 'ACTION_NIL'
		at the current state of the game.

		@param sso Observation of the current state of the game.
		@return True if the player can act, False otherwise.
		"""

		if self.game_playing == 'Catapults':
			bat_avatar_types = (11,12,13,14) # If the avatar type is one of these, the agent is a bat and cannot act
											 # (since it's in mid air)

			if sso.avatarType in bat_avatar_types:
				return False
			else:
				return True

		else:
			return True 


	def get_sso_subgoal_hash(self, sso, chosen_subgoal):
		"""
		Given the current state of the game and the chosen subgoal, it returns a hash. 
		Two different sso's will have the same hash if their observation matrices contain 
		the same observations in their first position of each cell.

		@param sso Observation of the current state of the game.
		@param chosen_subgoal Selected subgoal in the form of (x, y) coordinates.
		@return The hash of the (sso, chosen_subgoal) pair.
		"""
		obs_matrix = []
		obs = sso.observationGrid
		X_MAX = sso.observationGridNum
		Y_MAX = sso.observationGridMaxRow

		for y in range(Y_MAX):
			for x in range(X_MAX):
				observation = obs[x][y][0]

				if observation is None:
					obs_matrix.append(-1)
				else:
					obs_matrix.append(observation.itype)

		# Append the chosen subgoal to the matrix
		obs_matrix.append(chosen_subgoal)           

		return hash(tuple(obs_matrix))


	def encode_game_state(self, obs_grid, goal_pos):
		"""
		Transforms the game state (sso) from a matrix of observations to a matrix in which each
		position is one-hot encoded. If there are more than one observation at the same position,
		both are encoded. If there are no observations at a position, the resulting encoding
		is an array full of zeroes

		@param obs_grid Matrix of observations
		@param goal_pos (x, y) position (not as pixels but as grid) of the selected goal  
		"""

		# Dictionaries that maps itype (key) to one-hot array position to write a 1
		# e.g. 4 : 2 = [0 0 1 0 0 0 0 0 0]
		# None (empty tiles) objects are assigned an array full of zeroes

		# Boulder Dash
		# Itypes: 0, 1, 4, 5, 6, 7 (10 and 11 correspond to enemies)
		encode_dict_boulderdash = {
			0 : 0,
			1 : 1,
			4 : 2,
			5 : 3,
			6 : 4,
			7 : 5
		}

		# Catapults
		# Itypes: 0, 3, 5, 6, 7, 8, 9, 15
		encode_dict_catapults = {
			0 : 0,
			3 : 1,
			5 : 2,
			6 : 3,
			7 : 4,
			8 : 5,
			9 : 6,
			15 : 7
		}

		# IceAndFire
		# Itypes: 0, 1, 3, 4, 5, 6, 8, 9, 10
		encode_dict_iceandfire = {
			0 : 0,
			1 : 1,
			3 : 2,
			4 : 3,
			5 : 4,
			6 : 5,
			8 : 6,
			9 : 7,
			10 : 8
		}

		if self.game_playing=='BoulderDash':
			encode_dict = encode_dict_boulderdash
		elif self.game_playing=='IceAndFire':
			encode_dict = encode_dict_iceandfire
		else:
			encode_dict = encode_dict_catapults

		one_hot_length = len(encode_dict.keys())+1 # 1 extra position to represent the subgoal

		num_cols = len(obs_grid)
		num_rows = len(obs_grid[0])

		# The image representation is by rows and columns, instead of (x, y) pos of each pixel
		# Row -> y
		# Col -> x
		one_hot_grid = np.zeros((num_rows, num_cols, one_hot_length), np.int8)

		# Encode the grid
		for x in range(num_cols):
			for y in range(num_rows):
				for obs in obs_grid[x][y]:
					if obs is not None: # Ignore Empty tiles
						if not (self.game_playing == 'BoulderDash' and obs.itype == 3): # Ignore pickage images (those that correspond to ACTION_USE) in BoulderDash
							this_pos = encode_dict[obs.itype]
							one_hot_grid[y][x][this_pos] = 1

		# Encode the goal position within that grid
		one_hot_grid[goal_pos[1]][goal_pos[0]][one_hot_length-1] = 1

		return one_hot_grid

	def encode_game_state_all_subgoals(self, obs_grid, goal_pos_list):
		"""
		This method works the same as encode_game_state but returns a different one-hot
		matrix for each goal in goal_pos_list.

		@param obs_grid Matrix of observations
		@param goal_pos_list List in which every element corresponds to the (x, y) position 
							 (not as pixels but as grid) of a goal  
		"""

		# Dictionaries that maps itype (key) to one-hot array position to write a 1
		# e.g. 4 : 2 = [0 0 1 0 0 0 0 0 0]
		# None (empty tiles) objects are assigned an array full of zeroes

		# Boulder Dash
		# Itypes: 0, 1, 4, 5, 6, 7 (10 and 11 correspond to enemies)
		encode_dict_boulderdash = {
			0 : 0,
			1 : 1,
			4 : 2,
			5 : 3,
			6 : 4,
			7 : 5
		}

		# Catapults
		# Itypes: 0, 3, 5, 6, 7, 8, 9, 15
		encode_dict_catapults = {
			0 : 0,
			3 : 1,
			5 : 2,
			6 : 3,
			7 : 4,
			8 : 5,
			9 : 6,
			15 : 7
		}

		# IceAndFire
		# Itypes: 0, 1, 3, 4, 5, 6, 8, 9, 10
		encode_dict_iceandfire = {
			0 : 0,
			1 : 1,
			3 : 2,
			4 : 3,
			5 : 4,
			6 : 5,
			8 : 6,
			9 : 7,
			10 : 8
		}

		if self.game_playing=='BoulderDash':
			encode_dict = encode_dict_boulderdash
		elif self.game_playing=='IceAndFire':
			encode_dict = encode_dict_iceandfire
		else:
			encode_dict = encode_dict_catapults

		one_hot_length = len(encode_dict.keys())+1 # 1 extra position to represent the subgoal

		num_cols = len(obs_grid)
		num_rows = len(obs_grid[0])

		# The image representation is by rows and columns, instead of (x, y) pos of each pixel
		# Row -> y
		# Col -> x
		one_hot_grid = np.zeros((num_rows, num_cols, one_hot_length), np.int8)

		# Encode the grid
		for x in range(num_cols):
			for y in range(num_rows):
				for obs in obs_grid[x][y]:
					if obs is not None: # Ignore Empty tiles
						if not (self.game_playing == 'BoulderDash' and obs.itype == 3): # Ignore pickage images (those that correspond to ACTION_USE) in BoulderDash
							this_pos = encode_dict[obs.itype]
							one_hot_grid[y][x][this_pos] = 1

		# Repeat that one_hot_grid along a first, new numpy axis, so that there is a different
		# one_hot_grid for each subgoal
		one_hot_grid_array = np.repeat(one_hot_grid[np.newaxis, :, :, :], len(goal_pos_list), axis=0)

		# Encode a goal position in each grid
		for ind, goal_pos in enumerate(goal_pos_list):
			one_hot_grid_array[ind][goal_pos[1]][goal_pos[0]][one_hot_length-1] = 1

		return one_hot_grid_array

	def get_best_subgoals(self, sso, possible_subgoals):
		"""
		Returns a list with the subgoals in "possible_subgoals" ordered according to their
		corresponding Q-values obtained with the trained model. The first element of the list
		corresponds to the best subgoal, according to the predicted Q-value.

		@param obs_grid Current state of the game (instance of SerializableStateObservation class)
		@param possible_subgoals List of possible subgoals to choose from. Each element
								 corresponds to a (pos_x, pos_y) pair.
		"""

		"""
		obs_grid = sso.observationGrid

		best_plan_length = 1000000000.0
		best_subgoal = (-1, -1)

		for curr_subgoal in possible_subgoals:
			# Encode input for the network using current state and curr_subgoal
			one_hot_matrix = self.encode_game_state(obs_grid, curr_subgoal)

			# Use the model to predict plan length for curr_subgoal
			plan_length = self.model.predict(one_hot_matrix)

			# See if this is the best plan up to this point
			if plan_length < best_plan_length:
				best_plan_length = plan_length
				best_subgoal = curr_subgoal

		return best_subgoal"""

		observationGrid = sso.observationGrid

		# Retrieve resources the agent has at the sso state
		# agent_resources = self.get_agent_resources(sso)

		# Encode every subgoal as its corresponding one_hot_grid
		one_hot_grid_array = self.encode_game_state_all_subgoals(observationGrid, possible_subgoals)

		# Encode the agent resources to that every one_hot grid has the same resources
		# associated
		# agent_resources_array = np.repeat([agent_resources], len(possible_subgoals), axis=0)

		# Predict the Q_values for all the possible subgoals
		Q_values = self.model.predict_batch(one_hot_grid_array)

		# Order the subgoals according to their Q_values

		# Order this list according to the first element (Q_values), small first big last
		sorted_zip_list = sorted(zip(Q_values, possible_subgoals))

		# Obtain the ordered subgoals according to the Q_values
		ordered_subgoals = [goal for _, goal in sorted_zip_list]

		#print("----------------")
		#print("subgoals:", possible_subgoals)
		#print("Q_values:", Q_values)
		#print("Ordered:", ordered_subgoals)
		#print("----------------")

		return ordered_subgoals

	def order_subgoals_by_plan_length(self, sso, possible_subgoals):
		"""
		Returns a list with the subgoals ordered by plan length (the first one is the subgoal with the
		shortest valid plan). If for a given subgoal the planner can't find a valid plan, that subgoal
		is not added to the @ordered_subgoals list.
		This method DOES NOT measure planning time.

		@param obs_grid Current state of the game (instance of SerializableStateObservation class)
		@param possible_subgoals List of possible subgoals to choose from. Each element
								 corresponds to a (pos_x, pos_y) pair.
		"""

		# If the game is IceAndFire, check how many types of boots the agent has
		if self.game_playing == 'IceAndFire': 
			boots_resources = self.get_boots_resources(sso)
		else:
			boots_resources = []

		# For each subgoal, try to find a valid plan

		valid_subgoals = [] # List with all the subgoals for which a valid plan exists
		valid_subgoals_plan_lengths = [] # Plan length associated with each element of valid_subgoals

		for curr_subgoal in possible_subgoals:
			# Call the planner
			curr_plan = self.search_plan(sso, curr_subgoal, boots_resources)

			# If the subgoal is not valid (plan length == 0) do not add it to the list
			if len(curr_plan) > 0: 
				valid_subgoals.append(curr_subgoal)
				valid_subgoals_plan_lengths.append(len(curr_plan))

		# Order the subgoals according to their plan lengths

		# Order this list according to the first element (plan lengths), small first big last
		sorted_zip_list = sorted(zip(valid_subgoals_plan_lengths, valid_subgoals))

		# Obtain the ordered subgoals according to the plan lengths
		ordered_subgoals = [goal for _, goal in sorted_zip_list]

		return ordered_subgoals


	def get_min_Q_value(self, sso):
		"""
		This method is used to compute the Q-target. It calculates the minimum Q-value associated
		with the state sso.
		It employs the technique known as Double DQNs.
		It first uses the DQN network to select the best action (the one with the minimum
		predicted Q-value) to take in the state sso.
		It then predicts the Q-value associated with taking that action in state sso using
		the target network.
		This way it decouples action selection from Q-value generation.

		If sso is 'None' (corresponds to an end state), the Q-value is 0.

		@param sso Game state for which to calculate the Q-value.
		"""

		# Check if sso is a terminal state (end of level)
		if sso is None:
			return 0

		observationGrid = sso.observationGrid

		# Retrieve resources the agent has at the sso state
		# agent_resources = self.get_agent_resources(sso)

		# Retrieve subgoals list (list of subgoals positions)
		subgoals = self.get_subgoals_positions(sso)

		# Encode every subgoal as its corresponding one_hot_grid
		one_hot_grid_array = self.encode_game_state_all_subgoals(observationGrid, subgoals)

		# Encode the agent resources to that every one_hot grid has the same resources
		# associated
		# agent_resources_array = np.repeat([agent_resources], len(subgoals), axis=0)

		# Predict the Q_values for all the subgoals
		Q_values = self.target_network.predict_batch(one_hot_grid_array)

		# Get the min Q_value
		min_Q_val = np.min(Q_values)

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


	def search_plan(self, sso, goal, other_predicates):
		"""
		Method used to search a plan to a goal.

		@param sso State observation.
		@param goal (x, y) coordinates of the goal to be reached.
		@param other_predicates List of predicates to be appended to the ones
								of the current state. If the game is not
								IceAndFire, it will be an empty list [].
		@return Returns a plan to the current goal.
		"""

		x_goal, y_goal = goal # Separate the goal into its x,y coordinates

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


		# Add <ACTION_NIL> after each <ACTION_USE>

		new_plan = []

		for action in plan:
			if action == 'ACTION_USE':
				new_plan.append('ACTION_USE')
				new_plan.append('ACTION_NIL')
			else:
				new_plan.append(action)

		return new_plan


	def save_dataset(self, path, size_path=None):
		"""
		Uses the pickle module to save the experience replay to a file.
		It also stores the size of the dataset in the file given by
		size_path.

		@param path Path of the file
		@param size_path Path of the file containing the datasets sizes. If
						 it's None, then the size isn't saved.
		@param header If True, the first line of the file contains the number
			   of samples of the dataset.
		"""

		print("\nSaving experience replay...")

		with open(path, 'wb') as file:
			pickle.dump(self.memory, file)

		with open(size_path, 'a') as file:
			file.write("{}: {}\n".format(path, len(self.memory)))

		print("Saving finished!")


	def load_dataset(self, folder, game, num_levels=20, write_loaded_datasets=True, seed=None, max_samples_per_level=500):
		"""
		Uses the pickle module to load the previously saved experience replay.

		@folder Folder where the datasets are located (without '/')
		@game Game whose datasets to load
		@num_levels The number of levels whose datasets to load
		@write_loaded_datasets If True, the names of the loaded datasets are
							   written in the 'loaded_datasets.txt' file
		@seed Seed for level selection. If None, a random one is used.
		@max_samples_per_level If a level has more than this number of samples, only load the first "max_samples_per_level"
		"""

		# Delete current experience replay
		del self.memory[:]

		# Use glob to get all the different existing datasets in the folder
		# Use sorted to order them by name
		datasets = sorted(glob.glob('{}/dataset_{}*'.format(folder, game)))

		# Use a seed for repetibility
		if seed is not None:
			random.seed(seed)

		# Choose "num_levels" levels randomly
		selected_datasets = random.sample(datasets, k=num_levels)

		# Load each dataset
		total_num_samples = 0

		for dataset_path in selected_datasets:
			# Load the dataset and append the samples to memory
			with open(dataset_path, 'rb') as file:
				curr_dataset = pickle.load(file)

				if len(curr_dataset) > max_samples_per_level:
					curr_dataset = curr_dataset[:max_samples_per_level]

				total_num_samples += len(curr_dataset) # Store the number of samples
				self.memory.extend(curr_dataset)

			print("> {} loaded.".format(dataset_path))

		# Write the loaded datasets in a file
		if write_loaded_datasets:
			with open('loaded_datasets.txt', 'w') as file:
				for dataset_path in selected_datasets:
					file.write(dataset_path + '\n') 

		print(">> Loading finished!\n>> Number of samples loaded:", total_num_samples)
		

	def result(self, sso, elapsedTimer):
		print("> Level Finished")

		if self.EXECUTION_MODE == 'create_dataset':
			# Check if the current mem_sample is incomplete
			if len(self.mem_sample) == 3: 

				# If the player has lost the game, the plan of the mem_sample is invalid
				if sso.gameWinner == "PLAYER_LOSES":
					print("Incomplete mem_sample in result method - Player Loses")
					final_mem_sample = [self.mem_sample[0], self.mem_sample[1], self.num_actions_invalid_plan, None]

				# If the player has won, the plan is valid (the mem_sample is associated with a plan that completes the level)
				else:
					print("Incomplete mem_sample in result method - Player Wins")
					final_mem_sample = [self.mem_sample[0], self.mem_sample[1], self.mem_sample[2], None]

				# Add the final mem_sample of the level to the experience replay
				self.memory.append(final_mem_sample)
				

		if self.EXECUTION_MODE == 'test' and not self.is_training:

			# Check if the player hasn't been able to complete the level (timeout) or has died
			if sso.gameWinner == "PLAYER_LOSES":
				print("\n\nThe player has lost\n\n")
				self.num_incorrect_subgoals = -1 # This represents the agent has lost the game

			# Calculate average time per subgoal in the current level
			# mean_time_curr_level = self.total_time_curr_lv / self.num_calls_planner

			# Save the number of actions and the number of incorrect subgoals used to complete the 
			# current level to the output file
			test_output_file = "test_output.txt"

			with open(test_output_file, "a") as file:
				if self.goal_selection_mode == "best":
					file.write("{}-{} | {} | {} | {} | {} | {}\n".format(self.network_name, self.num_train_its_model,
					 self.game_playing, self.num_incorrect_subgoals, self.num_actions_lv,
					 self.total_time_planning_curr_lv, self.total_time_goal_selec_curr_lv))
				else:
					file.write("{} | {} | {} | {} | {}\n".format(self.network_name,
					 self.game_playing, self.num_incorrect_subgoals, self.num_actions_lv,
					 self.total_time_planning_curr_lv))

				if self.num_test_levels == 1: # For the last of the five val/test levels, write a linebreak
					file.write("\n-----------------------------------------------------------------------------------\n\n")

			# If the agent only needs to play one test level or has played the two test levels, then finish the execution
			if self.num_test_levels == 1 or self.playing_second_test_level:
				sys.exit()

			self.playing_second_test_level = True


		# Play a random level
		# Note: if mode is 'create_dataset', all three training levels should be the same!
		return -1


