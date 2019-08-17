import tensorflow as tf
import numpy as np

# Model used to choose next subgoal
# Given a pair (observation, subgoal) encoded as a one-hot observation matrix
# the model predicts the length of the plan asociated to that subgoal

class CNN:

	# Create CNN architecture
    def __init__(self, name="CNN", writer_name="CNN",
                 l1_num_filt = 2, l1_window = [4,4], l1_strides = [2,2],
                 padding_type = "SAME",
                 max_pool_size = [2, 2],
                 max_pool_str = [1, 1],
                 fc_num_units = 16, dropout_prob = 0.5,
                 learning_rate = 0.005):

        with tf.variable_scope(name):

            # --- Constants, Variables and Placeholders ---


            # Batch of inputs (game states, one-hot encoded)
            self.X = tf.placeholder(tf.float32, [None, 13, 26, 9], name="X") # type tf.float32 is needed for the rest of operations

            # Batch of outputs (correct predictions of number of actions)
            self.Y_corr = tf.placeholder(tf.float32, [None, 1], name="Y")
            
            # Placeholder for batch normalization
            # During training (big batches) -> true, during test (small batches) -> false
            self.is_training = tf.placeholder(tf.bool, name="is_training")

            # Learning Rate
            self.alfa = tf.constant(learning_rate)

            # Dropout Probability (probability of deactivation)
            self.dropout_prob = tf.constant(dropout_prob)
            

            # --- Architecture ---


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
                                         padding = padding_type,
                                         activation = tf.nn.relu,
                                         use_bias = True,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv1")
                        
            # Max pooling
            
            self.conv1 = tf.layers.max_pooling2d(inputs = self.conv1,
                                                pool_size = max_pool_size,
                                                strides = max_pool_str,
                                                padding = "VALID"
                                                )
            
             
            """
            Second convnet:
            """
            
            """
            self.conv2 = tf.layers.conv2d(inputs = self.conv1,
                                         filters = l2_num_filt,
                                         kernel_size = l2_window,
                                         strides = l2_strides,
                                         padding = padding_type,
                                         activation = tf.nn.relu,
                                         use_bias = True,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                         name = "conv2")
            
            # Max pooling
            
            self.conv2 = tf.layers.max_pooling2d(inputs = self.conv2,
                                                pool_size = [2, 2]
                                                strides = [2, 2]
                                                padding = "VALID",
                                                )
            """
            
            # Flatten output of conv layers
            
            self.flatten = tf.contrib.layers.flatten(self.conv1)
            
            
            # Fully connected layer
            
            self.fc = tf.layers.dense(inputs = self.flatten,
                                  units = fc_num_units,
                                  activation = tf.nn.relu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc")
            
            # Dropout
            
            self.fc = tf.layers.dropout(self.fc, rate=self.dropout_prob)
            
            # Output Layer
            
            self.output = tf.layers.dense(inputs = self.fc, 
                                          kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                          units = 1, 
                                          activation=None)
            
            # Train
            
            self.loss = tf.reduce_mean(tf.square(self.output - self.Y_corr), name="loss") # Quadratic loss
            
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.alfa, name="optimizer")
            
            # Mean and Variance Shift Operations needed for Batch Normalization
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            # Execute mean and variance updates of batch norm each training step
            with tf.control_dependencies(self.update_ops):
                self.train_op = self.optimizer.minimize(self.loss, name="train_op")
            

            # --- Summaries ---

            
            self.train_loss_sum = tf.summary.scalar('train_loss', self.loss) # Training loss
            self.test_loss_sum = tf.summary.scalar('test_loss', self.loss) # Validation loss
            
            self.writer = tf.summary.FileWriter("ModelLogs/" + writer_name)
            self.writer.add_graph(tf.get_default_graph())
            


        # --- Initialization ---


        # Create Session
        self.sess = tf.Session()

        # Initialize variables
        self.sess.run(tf.global_variables_initializer())


    # Closes the current tensorflow session and frees the resources.
    # Should be called at the end of the program
    def close_session(self):
        self.sess.close()

    # Predicts the associated y-value (plan length) for x (a (subgoal, game state) pair one-hot encoded)
    def predict(self, x):
        # Reshape x so that it has the shape of a one-element batch and can be fed into the placeholder
        x_resh = np.reshape(x, (1, 13, 26, 9))
        data_dict = {self.X : x_resh, self.is_training : False}

        prediction = self.sess.run(self.output, feed_dict=data_dict)

        return prediction

    # Execute num_it training steps using X, Y as the current batches. They must have the same number of elements
    def train(self, X, Y, num_it = 1):
        data_dict = {self.X : X, self.Y_corr : Y, self.is_training : True}

        for it in range(num_it):
            self.sess.run(self.train_op, feed_dict=data_dict)

    # Calculate Losses and store them as logs
    def save_logs(self, X_train, Y_train, X_test, Y_test, it):
        # Training Loss
        data_dict_train = {self.X : X_train, self.Y_corr : Y_train, self.is_training : True}

        train_loss_log = self.sess.run(self.train_loss_sum, feed_dict=data_dict_train)
        self.writer.add_summary(train_loss_log, it)

        # Validation Loss (uses validation dataset)
        data_dict_test = {self.X : X_test, self.Y_corr : Y_test, self.is_training : False}

        test_loss_log = self.sess.run(self.test_loss_sum, feed_dict=data_dict_test)
        self.writer.add_summary(test_loss_log, it)

    # Saves the model variables in the file given by 'path', so that it can be loaded next time
    def save_model(self, path = "./SavedModels/model.ckpt"):
        saver = tf.train.Saver()
        saver.save(self.sess, path)

    # Loads a model previously saved with 'save_model'
    def load_model(self, path = "./SavedModels/model.ckpt"):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)