import tensorflow as tf
import numpy as np

# Model used to choose next subgoal using Deep Q-Learning
# Given a pair (observation, subgoal) encoded as a one-hot observation matrix
# the model predicts the length of the plan that gets that subgoal and, then,
# completes the level

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
				 fc_num_units = [16, 1], dropout_prob = 0.5,
				 l2_regularization=0.0,
				 learning_rate = 0.005):

		self.variable_scope = name

		with tf.variable_scope(self.variable_scope):

			# --- Constants, Variables and Placeholders ---

			# Size of a sample, as rows x cols x (number of observations + 1)
			# It depends on the game being played
			self.sample_size = sample_size

			# Batch of inputs (game states + goals, one-hot encoded)
			X_shape = [None] # Batch size (needs to be known in advance for batch norm)
			X_shape.extend(self.sample_size) # e.g.: [None, 13, 26, 9]

			self.X = tf.placeholder(tf.float32, X_shape, name="X") # type tf.float32 is needed for the rest of operations

			# Q_target = R(s,a) + gamma * min Q(s', a') (s' next state after s, R(s,a) : plan length from state s to subgoal a)
			self.Q_target = tf.placeholder(tf.float32, [None, 1], name="Q_target")
			
			# Placeholder for batch normalization
			# During training (big batches) -> true, during test (small batches) -> false
			self.is_training = tf.placeholder(tf.bool, name="is_training")

			# Learning Rate
			self.alfa = tf.constant(learning_rate)

			# Dropout Probability (probability of deactivation)
			self.dropout_placeholder = tf.placeholder(tf.float32)
			self.dropout_prob = 0.5
			

			# --- Architecture ---


			"""
			Batch Normalization of inputs
			"""
			
			# Si quito esta normalización el training loss converge mucho peor!
			self.X_norm = tf.layers.batch_normalization(self.X, axis = 3, momentum=0.99, training=self.is_training)
			self.flatten = tf.contrib.layers.flatten(self.X_norm)


			# Intermediate layers

			self.fc_1 = tf.layers.dense(inputs = self.flatten,
					  units = 1024,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_1")
			self.fc_1 = tf.layers.dropout(self.fc_1, rate=self.dropout_placeholder)

			self.fc_2 = tf.layers.dense(inputs = self.fc_1,
					  units = 256,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_2")
			self.fc_2 = tf.layers.dropout(self.fc_2, rate=self.dropout_placeholder)

			self.fc_3 = tf.layers.dense(inputs = self.fc_2,
					  units = 128,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_3")
			self.fc_3 = tf.layers.dropout(self.fc_3, rate=self.dropout_placeholder)
	
			self.fc_4 = tf.layers.dense(inputs = self.fc_3,
					  units = 64,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_4")
			self.fc_4 = tf.layers.dropout(self.fc_4, rate=self.dropout_placeholder)
			
			self.fc_5 = tf.layers.dense(inputs = self.fc_4,
					  units = 32,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_5")
			# Don't use droput before output layer
			# self.fc_5 = tf.layers.dropout(self.fc_5, rate=self.dropout_placeholder)

			"""
			self.fc_6 = tf.layers.dense(inputs = self.fc_5,
					  units = 32,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_6")
			self.fc_6 = tf.layers.dropout(self.fc_6, rate=self.dropout_placeholder)
			

			self.fc_7 = tf.layers.dense(inputs = self.fc_6,
					  units = 16,
					  activation = tf.nn.relu,
					  kernel_initializer=tf.contrib.layers.xavier_initializer(),
					  name="fc_7")
			"""

			# Output Layer -> outputs the Q_value for the current (game state, subgoal) pair
			
			self.Q_val = tf.layers.dense(inputs = self.fc_5, 
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
	def predict(self, x):
		# Shape of a one-element batch
		x_shape = [1]
		x_shape.extend(self.sample_size) # e.g.: [1, 13, 26, 9]

		# Reshape x so that it has the shape of a one-element batch and can be fed into the placeholder
		x_resh = np.reshape(x, x_shape)
		data_dict = {self.X : x_resh, self.is_training : False, self.dropout_placeholder : 0.0}

		prediction = self.sess.run(self.Q_val, feed_dict=data_dict)

		return prediction

	# Predicts the associated y-value (plan length) for a batch of x ((subgoal, game state) pairs one-hot encoded)
	# Dropout is not activated
	def predict_batch(self, x):
		data_dict = {self.X : x, self.is_training : False, self.dropout_placeholder : 0.0}

		prediction = self.sess.run(self.Q_val, feed_dict=data_dict)

		return prediction

	# Execute num_it training steps using X, Y (Q_targets) as the current batches. They must have the same number of elements
	# Dropout is activated
	def train(self, X, Y, num_it = 1):
		data_dict = {self.X : X, self.Q_target : Y, self.is_training : True, self.dropout_placeholder : self.dropout_prob}

		for it in range(num_it):
			self.sess.run(self.train_op, feed_dict=data_dict)

	# Calculate Training Loss and store it as a log
	# Dropout is not activated
	def save_logs(self, X, Y, it):
		# Training Loss
		data_dict_train = {self.X : X, self.Q_target : Y, self.is_training : True, self.dropout_placeholder : 0.0}

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