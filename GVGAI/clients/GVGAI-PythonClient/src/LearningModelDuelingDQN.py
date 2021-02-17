import tensorflow as tf
import numpy as np

# Model used to choose next subgoal using Deep Q-Learning
# Given a pair (observation, subgoal) encoded as a one-hot observation matrix
# the model predicts the length of the plan that gets that subgoal and, then,
# completes the level

# Dueling DQN is very hard to implement for this model

class DQNetwork:

	# Create CNN architecture
	def __init__(self, name="DQNetwork", writer_name="DQNetwork", create_writer = True,
				 sess=None,
				 sample_size=[13, 26, 9],
				 l1_num_filt = 2, l1_window = [4,4], l1_strides = [2,2],
				 l1_padding_type = "SAME",
				 l2_num_filt = 2, l2_window = [4,4], l2_strides = [2,2],
				 l2_padding_type = "SAME",
				 l3_num_filt = 2, l3_window = [4,4], l3_strides = [2,2],
				 l3_padding_type = "SAME",
				 l4_num_filt = 2, l4_window = [4,4], l4_strides = [2,2],
				 l4_padding_type = "SAME",
				 l5_num_filt = 2, l5_window = [4,4], l5_strides = [2,2],
				 l5_padding_type = "SAME",
				 l6_num_filt = 2, l6_window = [4,4], l6_strides = [2,2],
				 l6_padding_type = "SAME",
				 l7_num_filt = 2, l7_window = [4,4], l7_strides = [2,2],
				 l7_padding_type = "SAME",
				 l8_num_filt = 2, l8_window = [4,4], l8_strides = [2,2],
				 l8_padding_type = "SAME",
				 l9_num_filt = 2, l9_window = [4,4], l9_strides = [2,2],
				 l9_padding_type = "SAME",
				 l10_num_filt = 2, l10_window = [4,4], l10_strides = [2,2],
				 l10_padding_type = "SAME",
				 l11_num_filt = 2, l11_window = [4,4], l11_strides = [2,2],
				 l11_padding_type = "SAME",
				 l12_num_filt = 2, l12_window = [4,4], l12_strides = [2,2],
				 l12_padding_type = "SAME",
				 l13_num_filt = 2, l13_window = [4,4], l13_strides = [2,2],
				 l13_padding_type = "SAME",
				 l14_num_filt = 2, l14_window = [4,4], l14_strides = [2,2],
				 l14_padding_type = "SAME",
				 l15_num_filt = 2, l15_window = [4,4], l15_strides = [2,2],
				 l15_padding_type = "SAME",
				 l16_num_filt = 2, l16_window = [4,4], l16_strides = [2,2],
				 l16_padding_type = "SAME",
				 l17_num_filt = 2, l17_window = [4,4], l17_strides = [2,2],
				 l17_padding_type = "SAME",
				 l18_num_filt = 2, l18_window = [4,4], l18_strides = [2,2],
				 l18_padding_type = "SAME",
				 l19_num_filt = 2, l19_window = [4,4], l19_strides = [2,2],
				 l19_padding_type = "SAME",
				 l20_num_filt = 2, l20_window = [4,4], l20_strides = [2,2],
				 l20_padding_type = "SAME",
				 fc_num_units = [16, 1, 1, 1], dropout_prob = 0.0,
				 learning_rate = 0.005,
				 use_BN = True, game_playing="BoulderDash"):

		self.variable_scope = name

		with tf.variable_scope(self.variable_scope):

			# --- Constants, Variables and Placeholders ---

			# Size of a sample, as rows x cols x (number of observations + 1)
			# It depends on the game being played
			self.sample_size = sample_size

			# Batch of inputs (game states + goals, one-hot encoded)
			X_shape = [None]
			X_shape.extend(self.sample_size) # e.g.: [None, 13, 26, 9]

			# Needs to be the one-hot-matrix for ALL the actions present at the state
			self.X = tf.placeholder(tf.float32, X_shape, name="X") # type tf.float32 is needed for the rest of operations

			# Batch of agent resources
			self.Agent_res = tf.placeholder(tf.float32, [None, 3], name="Agent_res")

			# Q_target = R(s,a) + gamma * min Q(s', a') (s' next state after s, R(s,a) : plan length from state s to subgoal a)
			self.Q_target = tf.placeholder(tf.float32, [None, 1], name="Q_target")
			
			# Placeholder for batch normalization
			# During training (big batches) -> true, during test (small batches) -> false
			self.is_training = tf.placeholder(tf.bool, name="is_training")

			# Learning Rate
			self.alfa = tf.constant(learning_rate)

			# Dropout Probability (probability of deactivation)
			self.dropout_placeholder = tf.placeholder(tf.float32)
			self.dropout_prob = dropout_prob
			

			# --- Architecture ---

			# If the game is BoulderDash, always use Batch Normalization
			if game_playing == "BoulderDash":
				use_BN = True

			"""
			Batch Normalization of inputs
			"""
			
			self.X_norm = tf.layers.batch_normalization(self.X, axis = 3, momentum=0.99, training=self.is_training)

			
			"""
			First convnet:
			"""
			
			# Padding = "VALID" -> no padding, "SAME" -> padding to keep the output dimension the same as the input one
			
			self.conv1 = tf.layers.conv2d(inputs = self.X_norm,
										 filters = l1_num_filt,
										 kernel_size = l1_window,
										 strides = l1_strides,
										 padding = l1_padding_type,
										 activation = tf.nn.leaky_relu,
										 use_bias = True,
										 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
										 name = "conv1")

			# Batch Normalization
			if use_BN:
				self.conv1 = tf.layers.batch_normalization(self.conv1, axis = 3, momentum=0.99, training=self.is_training)
			 
			"""
			Second convnet:
			"""
			
			self.conv2 = tf.layers.conv2d(inputs = self.conv1,
										 filters = l2_num_filt,
										 kernel_size = l2_window,
										 strides = l2_strides,
										 padding = l2_padding_type,
										 activation = tf.nn.leaky_relu,
										 use_bias = True,
										 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
										 name = "conv2")
			
			# Batch Normalization

			if use_BN:
				self.conv2 = tf.layers.batch_normalization(self.conv2, axis = 3, momentum=0.99, training=self.is_training)
			
			"""
			Third convnet:
			"""

			if l3_num_filt != -1:
				self.conv3 = tf.layers.conv2d(inputs = self.conv2,
								 filters = l3_num_filt,
								 kernel_size = l3_window,
								 strides = l3_strides,
								 padding = l3_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv3")
				
				# Batch Normalization

				if use_BN:
					self.conv3 = tf.layers.batch_normalization(self.conv3, axis = 3, momentum=0.99, training=self.is_training)
			
			else: # Only use three layers if l3_num_filt == -1
				self.conv3 = self.conv2


			"""
			Fourth convnet:
			"""

			if l4_num_filt != -1:
				self.conv4 = tf.layers.conv2d(inputs = self.conv3,
								 filters = l4_num_filt,
								 kernel_size = l4_window,
								 strides = l4_strides,
								 padding = l4_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv4")
				
				# Batch Normalization
				if use_BN:
					self.conv4 = tf.layers.batch_normalization(self.conv4, axis = 3, momentum=0.99, training=self.is_training)
			
			else: # Only use four layers if l4_num_filt == -1
				self.conv4 = self.conv3

			"""
			Fifth convnet:
			"""

			if l5_num_filt != -1:
				self.conv5 = tf.layers.conv2d(inputs = self.conv4,
								 filters = l5_num_filt,
								 kernel_size = l5_window,
								 strides = l5_strides,
								 padding = l5_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv5")
				
				# Batch Normalization
				if use_BN:
					self.conv5 = tf.layers.batch_normalization(self.conv5, axis = 3, momentum=0.99, training=self.is_training)
			
			else: # Only use four layers if l5_num_filt == -1
				self.conv5 = self.conv4

			"""
			Sixth convnet:
			"""
			if l6_num_filt != -1:
				self.conv6 = tf.layers.conv2d(inputs = self.conv5,
								 filters = l6_num_filt,
								 kernel_size = l6_window,
								 strides = l6_strides,
								 padding = l6_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv6")
				
				# Batch Normalization
				if use_BN:
					self.conv6 = tf.layers.batch_normalization(self.conv6, axis = 3, momentum=0.99, training=self.is_training)
			
			else: # Only use five layers if l6_num_filt == -1
				self.conv6 = self.conv5

			if l7_num_filt != -1:
				self.conv7 = tf.layers.conv2d(inputs = self.conv6,
								 filters = l7_num_filt,
								 kernel_size = l7_window,
								 strides = l7_strides,
								 padding = l7_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv7")
				
				# Batch Normalization
				if use_BN:
					self.conv7 = tf.layers.batch_normalization(self.conv7, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv7 = self.conv6

			if l8_num_filt != -1:
				self.conv8 = tf.layers.conv2d(inputs = self.conv7,
								 filters = l8_num_filt,
								 kernel_size = l8_window,
								 strides = l8_strides,
								 padding = l8_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv8")
				
				# Batch Normalization
				if use_BN:
					self.conv8 = tf.layers.batch_normalization(self.conv8, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv8 = self.conv7


			if l9_num_filt != -1:
				self.conv9 = tf.layers.conv2d(inputs = self.conv8,
								 filters = l9_num_filt,
								 kernel_size = l9_window,
								 strides = l9_strides,
								 padding = l9_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv9")
				
				# Batch Normalization
				if use_BN:
					self.conv9 = tf.layers.batch_normalization(self.conv9, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv9 = self.conv8


			if l10_num_filt != -1:
				self.conv10 = tf.layers.conv2d(inputs = self.conv9,
								 filters = l10_num_filt,
								 kernel_size = l10_window,
								 strides = l10_strides,
								 padding = l10_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv10")
				
				# Batch Normalization
				if use_BN:
					self.conv10 = tf.layers.batch_normalization(self.conv10, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv10 = self.conv9


			if l11_num_filt != -1:
				self.conv11 = tf.layers.conv2d(inputs = self.conv10,
								 filters = l11_num_filt,
								 kernel_size = l11_window,
								 strides = l11_strides,
								 padding = l11_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv11")
				
				# Batch Normalization
				if use_BN:
					self.conv11 = tf.layers.batch_normalization(self.conv11, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv11 = self.conv10


			if l12_num_filt != -1:
				self.conv12 = tf.layers.conv2d(inputs = self.conv11,
								 filters = l12_num_filt,
								 kernel_size = l12_window,
								 strides = l12_strides,
								 padding = l12_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv12")
				
				# Batch Normalization
				if use_BN:
					self.conv12 = tf.layers.batch_normalization(self.conv12, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv12 = self.conv11


			if l13_num_filt != -1:
				self.conv13 = tf.layers.conv2d(inputs = self.conv12,
								 filters = l13_num_filt,
								 kernel_size = l13_window,
								 strides = l13_strides,
								 padding = l13_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv13")
				
				# Batch Normalization
				if use_BN:
					self.conv13 = tf.layers.batch_normalization(self.conv13, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv13 = self.conv12


			if l14_num_filt != -1:
				self.conv14 = tf.layers.conv2d(inputs = self.conv13,
								 filters = l14_num_filt,
								 kernel_size = l14_window,
								 strides = l14_strides,
								 padding = l14_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv14")
				
				# Batch Normalization
				if use_BN:
					self.conv14 = tf.layers.batch_normalization(self.conv14, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv14 = self.conv13


			if l15_num_filt != -1:
				self.conv15 = tf.layers.conv2d(inputs = self.conv14,
								 filters = l15_num_filt,
								 kernel_size = l15_window,
								 strides = l15_strides,
								 padding = l15_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv15")
				
				# Batch Normalization
				if use_BN:
					self.conv15 = tf.layers.batch_normalization(self.conv15, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv15 = self.conv14


			if l16_num_filt != -1:
				self.conv16 = tf.layers.conv2d(inputs = self.conv15,
								 filters = l16_num_filt,
								 kernel_size = l16_window,
								 strides = l16_strides,
								 padding = l16_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv16")
				
				# Batch Normalization
				if use_BN:
					self.conv16 = tf.layers.batch_normalization(self.conv16, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv16 = self.conv15


			if l17_num_filt != -1:
				self.conv17 = tf.layers.conv2d(inputs = self.conv16,
								 filters = l17_num_filt,
								 kernel_size = l17_window,
								 strides = l17_strides,
								 padding = l17_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv17")
				
				# Batch Normalization
				if use_BN:
					self.conv17 = tf.layers.batch_normalization(self.conv17, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv17 = self.conv16

			
			if l18_num_filt != -1:
				self.conv18 = tf.layers.conv2d(inputs = self.conv17,
								 filters = l18_num_filt,
								 kernel_size = l18_window,
								 strides = l18_strides,
								 padding = l18_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv18")
				
				# Batch Normalization
				if use_BN:
					self.conv18 = tf.layers.batch_normalization(self.conv18, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv18 = self.conv17


			if l19_num_filt != -1:
				self.conv19 = tf.layers.conv2d(inputs = self.conv18,
								 filters = l19_num_filt,
								 kernel_size = l19_window,
								 strides = l19_strides,
								 padding = l19_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv19")
				
				# Batch Normalization
				if use_BN:
					self.conv19 = tf.layers.batch_normalization(self.conv19, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv19 = self.conv18


			if l20_num_filt != -1:
				self.conv20 = tf.layers.conv2d(inputs = self.conv19,
								 filters = l20_num_filt,
								 kernel_size = l20_window,
								 strides = l20_strides,
								 padding = l20_padding_type,
								 activation = tf.nn.leaky_relu,
								 use_bias = True,
								 kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
								 name = "conv20")
				
				# Batch Normalization
				if use_BN:
					self.conv20 = tf.layers.batch_normalization(self.conv20, axis = 3, momentum=0.99, training=self.is_training)
			
			else: 
				self.conv20 = self.conv19


			# Flatten output of conv layers
			self.flatten = tf.contrib.layers.flatten(self.conv20)

			# Concatenate agent resources
			# self.flatten = tf.concat([self.flatten, self.Agent_res], 1)
			
			# <Value Network>

			# Fully connected layer 1

			self.value_fc_1 = tf.layers.dense(inputs = self.flatten,
								  units = fc_num_units[0],
								  activation = tf.nn.leaky_relu,
								  kernel_initializer=tf.contrib.layers.xavier_initializer(),
								  name="value_fc_1")

			# Fully connected layer 2
			# <Only if fc_num_units[1] != 1>

			if fc_num_units[1] != 1:
			  # Dropout 1
			  # If there is a single fc layer, don't use droput after it since it is right before
			  # the output layer
			  self.value_fc_1 = tf.layers.dropout(self.value_fc_1, rate=self.dropout_placeholder, name="Dropout_1")

			  self.value_fc_2 = tf.layers.dense(inputs = self.value_fc_1,
									units = fc_num_units[1],
									activation = tf.nn.leaky_relu,
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									name="value_fc_2")

			else:
			  self.value_fc_2 = self.value_fc_1
			
			# Fully connected layer 3
			# <Only if fc_num_units[2] != 1>
			if fc_num_units[2] != 1:
			  # Dropout 2
			  # If there is a single fc layer, don't use droput after it since it is right before
			  # the output layer
			  self.value_fc_2 = tf.layers.dropout(self.value_fc_2, rate=self.dropout_placeholder, name="Dropout_2")

			  self.value_fc_3 = tf.layers.dense(inputs = self.value_fc_2,
									units = fc_num_units[2],
									activation = tf.nn.leaky_relu,
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									name="value_fc_3")

			else:
			  self.value_fc_3 = self.value_fc_2

			# Fully connected layer 4
			# <Only if fc_num_units[3] != 1>
			if fc_num_units[3] != 1:
			  # Dropout 3
			  # If there is a single fc layer, don't use droput after it since it is right before
			  # the output layer
			  self.value_fc_3 = tf.layers.dropout(self.value_fc_3, rate=self.dropout_placeholder, name="Dropout_3")

			  self.value_fc_4 = tf.layers.dense(inputs = self.value_fc_3,
									units = fc_num_units[3],
									activation = tf.nn.leaky_relu,
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									name="value_fc_4")

			else:
			  self.value_fc_4 = self.value_fc_3

			# Output of value network
			self.value_output = tf.layers.dense(inputs = self.value_fc_4, 
										  kernel_initializer=tf.contrib.layers.xavier_initializer(),
										  units = 1, 
										  activation=None,
										  name="value_output")

			# <Advantage Network>

			# Fully connected layer 1

			self.advantage_fc_1 = tf.layers.dense(inputs = self.flatten,
								  units = fc_num_units[0],
								  activation = tf.nn.leaky_relu,
								  kernel_initializer=tf.contrib.layers.xavier_initializer(),
								  name="advantage_fc_1")

			# Fully connected layer 2
			# <Only if fc_num_units[1] != 1>

			if fc_num_units[1] != 1:
			  # Dropout 1
			  # If there is a single fc layer, don't use droput after it since it is right before
			  # the output layer
			  self.advantage_fc_1 = tf.layers.dropout(self.advantage_fc_1, rate=self.dropout_placeholder, name="Dropout_1")

			  self.advantage_fc_2 = tf.layers.dense(inputs = self.advantage_fc_1,
									units = fc_num_units[1],
									activation = tf.nn.leaky_relu,
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									name="advantage_fc_2")

			else:
			  self.advantage_fc_2 = self.advantage_fc_1
			
			# Fully connected layer 3
			# <Only if fc_num_units[2] != 1>
			if fc_num_units[2] != 1:
			  # Dropout 2
			  # If there is a single fc layer, don't use droput after it since it is right before
			  # the output layer
			  self.advantage_fc_2 = tf.layers.dropout(self.advantage_fc_2, rate=self.dropout_placeholder, name="Dropout_2")

			  self.advantage_fc_3 = tf.layers.dense(inputs = self.advantage_fc_2,
									units = fc_num_units[2],
									activation = tf.nn.leaky_relu,
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									name="advantage_fc_3")

			else:
			  self.advantage_fc_3 = self.advantage_fc_2

			# Fully connected layer 4
			# <Only if fc_num_units[3] != 1>
			if fc_num_units[3] != 1:
			  # Dropout 3
			  # If there is a single fc layer, don't use droput after it since it is right before
			  # the output layer
			  self.advantage_fc_3 = tf.layers.dropout(self.advantage_fc_3, rate=self.dropout_placeholder, name="Dropout_3")

			  self.advantage_fc_4 = tf.layers.dense(inputs = self.advantage_fc_3,
									units = fc_num_units[3],
									activation = tf.nn.leaky_relu,
									kernel_initializer=tf.contrib.layers.xavier_initializer(),
									name="advantage_fc_4")

			else:
			  self.advantage_fc_4 = self.advantage_fc_3			

			# Output of advantage network
			self.advantage_output = tf.layers.dense(inputs = self.advantage_fc_4, 
										  kernel_initializer=tf.contrib.layers.xavier_initializer(),
										  units = 1, 
										  activation=None,
										  name="advantage_output")


			# <Aggregation layer>

			# For each (s,a) pair, we obtain a different state value (self.value_output)
			# althought all of them should be the same -> we consider the average value
			# as the "correct" state value
			self.state_value = tf.reduce_mean(self.value_output, name="state_value")

			# Average Advantage
			self.average_advantage = tf.reduce_mean(self.advantage_output, name="average_advantage")

			# Advantage of the corresponding action


			# Output Layer -> outputs the Q_value for the current (game state, subgoal) pair
			
			self.Q_val = tf.layers.dense(inputs = self.fc_4, 
										  kernel_initializer=tf.contrib.layers.xavier_initializer(),
										  units = 1, 
										  activation=None,
										  name="Q_value")
			
			# Train
			
			# The loss is the difference between our predicted Q_values and the Q_targets
			# Sum(Qtarget - Q)^2
			self.loss = tf.reduce_mean(tf.square(self.Q_target - self.Q_val), name="loss")
			
			self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alfa, name="optimizer")
			
			# Mean and Variance Shift Operations needed for Batch Normalization
			self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

			# Execute mean and variance updates of batch norm each training step
			with tf.control_dependencies(self.update_ops):
				self.train_op = self.optimizer.minimize(self.loss, name="train_op")
			

			# --- Summaries ---

			if create_writer:
			  self.train_loss_sum = tf.summary.scalar('train_loss', self.loss) # Training loss
			  self.Q_val_sum = tf.summary.scalar('Q_val', tf.reduce_mean(self.Q_val))
			  self.Q_target_sum = tf.summary.scalar('Q_target', tf.reduce_mean(self.Q_target))
	  
			  self.writer = tf.summary.FileWriter("DQNetworkLogs/" + writer_name)
			  self.writer.add_graph(tf.get_default_graph())
			

		# --- Initialization ---

		if sess is None:
			# Create Session

			# Run on GPU
			
			# Needed for running on GPU
			gpu_options = tf.GPUOptions(allow_growth=True)
			self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
			
			# Run on CPU (it's faster if model and batch_size is small)               
			# self.sess = tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0}))
		else:
			# Reuse session passed as a parameter
			self.sess = sess

		# Initialize variables
		self.sess.run(tf.global_variables_initializer())


	# Closes the current tensorflow session and frees the resources.
	# Should be called at the end of the program
	def close_session(self):
		self.sess.close()

	# Predicts the associated y-value (plan length) for x (a (subgoal, game state) pair one-hot encoded)
	# Dropout is not activated
	def predict(self, x, agent_res):
		# Shape of a one-element batch
		x_shape = [1]
		x_shape.extend(self.sample_size) # e.g.: [1, 13, 26, 9]

		# Reshape x so that it has the shape of a one-element batch and can be fed into the placeholder
		x_resh = np.reshape(x, x_shape)

		# Reshape the list with the agent resources too
		agent_res_resh = np.reshape(agent_res, [1, 3])

		data_dict = {self.X : x_resh, self.Agent_res : agent_res_resh,
		 self.is_training : False, self.dropout_placeholder : 0.0}

		prediction = self.sess.run(self.Q_val, feed_dict=data_dict)

		return prediction

	# Predicts the associated y-value (plan length) for a batch of x ((subgoal, game state) pairs one-hot encoded)
	# Dropout is not activated
	def predict_batch(self, x, agent_res):
		data_dict = {self.X : x, self.Agent_res : agent_res,
		 self.is_training : False, self.dropout_placeholder : 0.0}

		prediction = self.sess.run(self.Q_val, feed_dict=data_dict)

		return prediction

	# Execute num_it training steps using X, Y (Q_targets) as the current batches. They must have the same number of elements
	# Dropout is activated
	def train(self, X, Agent_res, Y, num_it = 1):
		data_dict = {self.X : X, self.Agent_res : Agent_res,
		 self.Q_target : Y, self.is_training : True, self.dropout_placeholder : self.dropout_prob}

		for it in range(num_it):
			self.sess.run(self.train_op, feed_dict=data_dict)

	# Calculate Training Loss and store it as a log
	# Dropout is not activated
	def save_logs(self, X, Agent_res, Y, it):
		# Training Loss
		data_dict_train = {self.X : X, self.Agent_res : Agent_res,
		 self.Q_target : Y, self.is_training : True, self.dropout_placeholder : 0.0}

		train_loss_log, Q_val_log, Q_target_log = self.sess.run([self.train_loss_sum, self.Q_val_sum,
		 self.Q_target_sum], feed_dict=data_dict_train)

		self.writer.add_summary(train_loss_log, it)
		self.writer.add_summary(Q_val_log, it)
		self.writer.add_summary(Q_target_log, it)

	# Saves the model variables in the file given by 'path', so that it can be loaded next time
	def save_model(self, path = "./SavedModels/DQmodel.ckpt", num_it = None):
		saver = tf.train.Saver(
		  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.variable_scope)) # Only save the variables within this variable scope
		
		if num_it is None:
		  saver.save(self.sess, path)
		else:
		  saver.save(self.sess, path, global_step = num_it)

	# Loads a model previously saved with 'save_model'
	def load_model(self, path = "./SavedModels/DQmodel.ckpt", num_it = None):
		saver = tf.train.Saver(
		  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.variable_scope)) # Only load the variables within this variable scope
		
		if num_it is None:
		  saver.restore(self.sess, path)
		else:
		  saver.restore(self.sess, path + '-' + str(num_it))

	# Runs update_ops to update the current weights. This method is used to update the target network's weights
	# every tau steps
	def update_weights(self, update_ops):
		self.sess.run(update_ops)