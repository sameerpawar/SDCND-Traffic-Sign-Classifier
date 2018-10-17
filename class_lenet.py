#*******************************************************************************#
# Sameer Pawar, Oct-2018                                                          
# LeNet++ implementation in TensorFlow                              
# - unlike original LeNet, LeNet++ accepts color 32*32 images and any number of classes (< 84)
#*******************************************************************************#
import time
import tensorflow as tf
from tensorflow.contrib.layers import flatten, batch_norm
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from img_lib import normalize_image, histogram_equalize_image

class LeNet:

    def __init__(self, n_classes):        
        self.n_classes = n_classes        
        self.max_validation_accuracy = 0.95
        self.max_id = -1
        self.initializer = tf.contrib.layers.xavier_initializer() 
        self.activation_fun = None
        self.optimizer = tf.train.AdamOptimizer()
        self.layers = {}

        # Define weights and biases for the graph
        self.weights = {
        'w_conv_0': tf.get_variable("w_conv_0", shape = [1, 1, 3, 1],       initializer = self.initializer),
        'w_conv_1': tf.get_variable("w_conv_1", shape = [5, 5, 1, 6],       initializer = self.initializer),
        'w_conv_2': tf.get_variable("w_conv_2", shape = [5, 5, 6, 16],      initializer = self.initializer),
        'w_fc_3':   tf.get_variable("w_fc_3",   shape = [400, 120],         initializer = self.initializer),
        'w_fc_4':   tf.get_variable("w_fc_4",   shape = [120, 84],          initializer = self.initializer),
        'w_fc_5':   tf.get_variable("w_fc_5",  shape = [84, self.n_classes],initializer = self.initializer)
        }

        self.biases = {
        'b_conv_0': tf.get_variable("b_conv_0", initializer = tf.zeros(1)),
        'b_conv_1': tf.get_variable("b_conv_1", initializer = tf.zeros(6)),
        'b_conv_2': tf.get_variable("b_conv_2", initializer = tf.zeros(16)),
        'b_fc_3':   tf.get_variable("b_fc_3",   initializer = tf.zeros(120)),
        'b_fc_4':   tf.get_variable("b_fc_4",   initializer = tf.zeros(84)),
        'b_fc_5':   tf.get_variable("b_fc_5", initializer = tf.zeros(self.n_classes))
        }
        self.keep_prob = np.ones(len(self.weights))
        
        self.x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self.y = tf.placeholder(tf.int32, (None))
        self.batch_norm  = tf.placeholder(tf.bool)
        self.keep_probabilities = tf.placeholder(tf.float32, shape = (len(self.weights)))

        #********************************************************************************************
        # Define compute nodes in a graph
        #********************************************************************************************
        self.one_hot_y          = tf.one_hot(self.y, self.n_classes)
        self.logits             = self.get_logits(self.x, self.keep_probabilities, self.batch_norm)
        self.cross_entropy      = tf.nn.softmax_cross_entropy_with_logits(labels=self.one_hot_y, logits=self.logits)
        self.loss_operation     = tf.reduce_mean(self.cross_entropy)    
        self.training_operation = self.optimizer.minimize(self.loss_operation)        
        #********************************************************************************************
    
    def init_dropout_probabilities(self, dropout_probabilities):        
        if 'keep_conv_0' in dropout_probabilities:
            self.keep_prob[1] = dropout_probabilities['keep_conv_0']
        if 'keep_conv_1' in dropout_probabilities:
            self.keep_prob[2] = dropout_probabilities['keep_conv_1']
        if 'keep_conv_2' in dropout_probabilities:
            self.keep_prob[3] = dropout_probabilities['keep_conv_2']
        if 'keep_fc_3' in dropout_probabilities:
            self.keep_prob[4] = dropout_probabilities['keep_fc_3']
        if 'keep_fc_4' in dropout_probabilities:
            self.keep_prob[5] = dropout_probabilities['keep_fc_4']
        
        return


    def one_hot_encoding(self, labels):
        b = np.zeros((len(labels), self.n_classes))
        b[np.arange(len(labels)), labels] = 1
        return b

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(
            x,
            ksize=[1, k, k, 1],
            strides=[1, k, k, 1],
            padding='SAME')
    
    def activation(self, x, BN = False):    
        if self.activation_fun == 'relu':
            output = tf.nn.relu(x)
        elif self.activation_fun == 'elu':
            output = tf.nn.elu(x)
        elif self.activation_fun == 'lrelu':
            output = tf.nn.leaky_relu(x, alpha = 0.1)
        else:
            # default activation is 'linear'
            output = x

        if BN:
            return batch_norm(output)
        else:
            return output
    
    

    def conv2d(self, x, W, b, strides=1):
        return tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='VALID') + b

    
    def preprocess_data(self, X_data):
        return np.array([normalize_image(histogram_equalize_image(image)) for image in X_data]).reshape(X_data.shape[0], 32,32,-1)
       
    
    def get_logits(self, x, keep_probabilities, batch_norm_flag):                
        # conv-Layer 1 - 32*32*3 -> 32*32*1
        # maps 3 color channels to one generalized grayscale channel
        self.layers['l_1'] = self.conv2d(x, W = self.weights['w_conv_0'], b = self.biases['b_conv_0'])
        self.layers['l_1'] = tf.nn.dropout(self.layers['l_1'], self.keep_probabilities[1])

        # conv-Layer 2 - 32*32*1 -> 28*28*6->14*14*6
        self.layers['l_2'] = self.conv2d(self.layers['l_1'], self.weights['w_conv_1'], self.biases['b_conv_1'])
        self.layers['l_2'] = self.maxpool2d(self.layers['l_2'])
        self.layers['l_2'] = tf.cond(batch_norm_flag, 
                                     lambda: self.activation(self.layers['l_2'], BN = True), 
                                     lambda: self.activation(self.layers['l_2'], BN = False)
                                    )
        self.layers['l_2'] = tf.nn.dropout(self.layers['l_2'], keep_probabilities[2])        

        # conv-Layer 3 - 14*14*6 -> 10*10*16->5*5*16
        self.layers['l_3'] = self.conv2d(self.layers['l_2'], self.weights['w_conv_2'], self.biases['b_conv_2'])
        self.layers['l_3'] = self.maxpool2d(self.layers['l_3'])
        self.layers['l_3'] = tf.cond(batch_norm_flag, 
                                     lambda: self.activation(self.layers['l_3'], BN = True), 
                                     lambda: self.activation(self.layers['l_3'], BN = False)
                                    )
        self.layers['l_3'] = tf.nn.dropout(self.layers['l_3'], keep_probabilities[3])        

        # FC-Layer 4 - 5*5*16-> 400->120
        self.layers['l_4'] = tf.matmul(flatten(self.layers['l_3']), self.weights['w_fc_3']) + self.biases['b_fc_3']
        self.layers['l_4'] = tf.cond(batch_norm_flag, 
                                     lambda: self.activation(self.layers['l_4'], BN = True), 
                                     lambda: self.activation(self.layers['l_4'], BN = False)
                                    )
        self.layers['l_4'] = tf.nn.dropout(self.layers['l_4'], keep_probabilities[4])        
        
        # FC-Layer 5 - 120-> 84
        self.layers['l_5'] = tf.matmul(self.layers['l_4'], self.weights['w_fc_4']) + self.biases['b_fc_4']
        self.layers['l_5'] = tf.cond(batch_norm_flag, 
                                     lambda: self.activation(self.layers['l_5'], BN = True), 
                                     lambda: self.activation(self.layers['l_5'], BN = False)
                                    )
        self.layers['l_5'] = tf.nn.dropout(self.layers['l_5'], keep_probabilities[5])

        # FC-Layer 6 - 84 -> 43 (n_classes)
        self.layers['l_6'] = tf.matmul(self.layers['l_5'], self.weights['w_fc_5']) + self.biases['b_fc_5']

        
        logits = self.layers['l_6']

        return logits

    def compile(self, activation_function = 'relu', optimizer = None, loss=None, metrics = None):
        """
        #************************************************************************#
            - activation_function = 'relu', 'elu', 'lrelu' 
            - optimizer: String (name of optimizer) or optimizer instance.
            - loss: String (name of objective function) or objective function.
            - metrics:  List of metrics to be evaluated by the model during training 
            and testing. Typically you will use  metrics=['accuracy']. 
            Also can pass multiple metrics such as 'accuracy', 
            'Precision-recall-f1score', etc.
        #************************************************************************#            
        """
        self.activation_fun = activation_function

        if optimizer is None:
            self.optimizer = tf.train.AdamOptimizer()
        else:
            self.optimizer = optimizer
        
        if loss is None:
            self.loss = 'cross-entropy'
        else:
            self.loss = loss

        if metrics is None:
            self.metrics = 'accuracy'    
        else:
            self.metrics = metrics # dictionary of different metrics
           
    def fit(self, x=None, y=None, generator = None, batch_size=None, epochs=1, 
    validation_split=0.0, validation_data=None, dropout_probabilities = None,
    save_trained_weights = None, verbose = False):
        """
        #************************************************************************#
            - x: input features
            - y: labels
            - generator: function that takes in a batch of data (x, y) and returns a batch of data of same size but with transformed/augmented data
            - batch_size
            - epochs
            - validation_split 
            - validation_data: dictionary {'features': x_valid, 'labels': y_valid}
            - dropout_probabilities: dictionary of Keep probabilities with key = 'keep_conv/fc_id'
        #************************************************************************#
        """
        self.training_loss          = np.zeros(epochs)
        self.training_accuracy      = np.zeros(epochs)
        self.training_precision     = np.zeros((self.n_classes, epochs))
        self.training_recall        = np.zeros((self.n_classes, epochs))
        self.training_F1score       = np.zeros((self.n_classes, epochs))

        self.validation_loss        = np.zeros(epochs)
        self.validation_accuracy    = np.zeros(epochs)
        self.validation_precision   = np.zeros((self.n_classes, epochs))
        self.validation_recall      = np.zeros((self.n_classes, epochs))
        self.validation_F1score     = np.zeros((self.n_classes, epochs))


        # Get training and validation data        
        if validation_data is not None:
            x_valid, y_valid = validation_data['features'], validation_data['labels']
        elif validation_split > 0:
            x, x_valid, y, y_valid = train_test_split(x, y, train_size=1-validation_split)
        else:
            x, x_valid, y, y_valid = train_test_split(x, y, train_size=1-0.2)
            
        # generate pre-processed training and validation sets for evaluations.
        x_train_preprocessed, y_train_preprocessed = (self.preprocess_data(x), y)
        x_valid_preprocessed, y_valid_preprocessed = (self.preprocess_data(x_valid), y_valid)
        
        # Read the keep-probabilities for dropout
        if dropout_probabilities is not None:
            self.init_dropout_probabilities(dropout_probabilities)

        with tf.Session() as sess:      
            sess.run(tf.global_variables_initializer())            
            saver = tf.train.Saver()
            for i in range(epochs):    
                if verbose:      
                    print("running EPOCH {}".format(i+1))
                    t1 = time.time()
                x, y = shuffle(x, y)
                for offset in range(0, x.shape[0], batch_size):
                    end = np.minimum(offset + batch_size, x.shape[0])
                    if generator is None:
                        batch_x, batch_y = self.preprocess_data(x[offset:end]), y[offset:end]
                    else:
                        batch_x, batch_y = self.preprocess_data(generator(x[offset:end])), y[offset:end]
                            
                    
                    sess.run(self.training_operation, feed_dict={self.x: batch_x, self.y: batch_y, 
                                                                 self.keep_probabilities: self.keep_prob,
                                                                 self.batch_norm: True})
            
                # Compute the metrics for each epoch and log the results
                
                
                training_results   = self.evaluate(x_train_preprocessed, y_train_preprocessed)
                validation_results = self.evaluate(x_valid_preprocessed, y_valid_preprocessed)                

                self.training_accuracy[i]    = training_results['accuracy']
                self.training_loss[i]        = training_results['loss']        
                self.training_precision[:,i] = training_results['precision']
                self.training_recall[:,i]    = training_results['recall']
                self.training_F1score[:,i]   = training_results['F1score']

                self.validation_accuracy[i]    = validation_results['accuracy']
                self.validation_loss[i]        = validation_results['loss']        
                self.validation_precision[:,i] = validation_results['precision']
                self.validation_recall[:,i]    = validation_results['recall']
                self.validation_F1score[:,i]   = validation_results['F1score']    

                if self.validation_accuracy[i]  > self.max_validation_accuracy:
                    self.max_validation_accuracy = self.validation_accuracy[i]
                    self.max_id                  = i
                    if save_trained_weights is not None:
                        saver.save(sess, save_trained_weights)

                if verbose: 
                    print("time for one epoch {:.2f}".format(time.time()-t1))
                    print("training   accuracy = {:.3f}".format(training_results['accuracy']))
                    print("validation accuracy = {:.3f}".format(validation_results['accuracy']))
                    print("training   loss = {:.3f}".format(training_results['loss']))
                    print("validation loss = {:.3f}".format(validation_results['loss']))
                    print("max validation accuracy {:.3f} is at epoch {}".format(self.max_validation_accuracy, self.max_id + 1))

        return

    def evaluate(self, x=None, y=None):
        x, y = shuffle(x, y, random_state = 29)
        batch_size = 1024
        num_examples  = x.shape[0]
        sess = tf.get_default_session()
        total_TP = np.zeros(self.n_classes, dtype=np.float32)
        total_FP = np.zeros(self.n_classes, dtype=np.float32)
        total_FN = np.zeros(self.n_classes, dtype=np.float32)    
        total_correct_predictions = 0
        total_loss = 0        
        error_list = []
        for offset in range(0, num_examples, batch_size):
            end = np.minimum(offset + batch_size, num_examples)
            batch_x, batch_y = x[offset:end], y[offset:end]
            batch_logits, batch_loss = sess.run([self.logits, self.loss_operation], 
                                                    feed_dict = {self.x: batch_x, 
                                                            self.y: batch_y,
                                                            self.keep_probabilities: np.ones(len(self.weights)),
                                                            self.batch_norm: True}
                                                    )
                       
            predicted_labels = np.argmax(batch_logits, axis = 1)
            
            actuals      = self.one_hot_encoding(batch_y)                       
            predictions  = self.one_hot_encoding(predicted_labels)                
            total_TP += np.count_nonzero(predictions * actuals, axis=0)
            total_FP += np.count_nonzero(predictions * (actuals - 1), axis=0)
            total_FN += np.count_nonzero((predictions - 1) * actuals, axis=0)     
        
            
            total_correct_predictions += np.sum(np.equal(predicted_labels, batch_y))
            total_loss                += (batch_loss * batch_x.shape[0])

            predictions = np.array(predicted_labels).flatten()
            actuals     = np.array(batch_y).flatten()
            error_idx   = np.argwhere(np.not_equal(predictions, actuals)).ravel() + offset
            error_list.append(error_idx)        
        
        
        error_list = np.concatenate(error_list)
        accuracy  = total_correct_predictions/num_examples
        loss      = total_loss/num_examples
        precision = total_TP/(total_TP + total_FP + 1e-6)
        recall    = total_TP/(total_TP + total_FN + 1e-6)
        F1score   = 2 * precision * recall / (precision + recall + 1e-6)
        return {"loss": loss,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "F1score": F1score,
                "error_list": error_list
            }

    def get_results_per_epoch(self):
        return {'training_loss': self.training_loss, 
                'training_accuracy': self.training_accuracy,
                'validation_loss': self.validation_loss,
                'validation_accuracy': self.validation_accuracy
        }

    def predict(self, x, top_k = 1):
        sess                = tf.get_default_session()
        predicted_logits    = sess.run( self.logits, 
                                        feed_dict = {   self.x: x,
                                                        self.keep_probabilities: np.ones(len(self.weights)), 
                                                        self.batch_norm: False
                                                    }
                                        )
        softmax, class_Id   = sess.run(tf.nn.top_k(tf.nn.softmax(predicted_logits), top_k))
        return softmax, class_Id