
##we will try regularization on the nn with gradient descent 
##also we add the RElu activation

import numpy as np
import tensorflow as tf
import pickle


pickle_file = 'notMNIST.pickle'



with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  

image_size = 28
num_labels = 10

def reformat(dataset, labels): #1-hot encoding
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
  print(labels)
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  print(labels)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)


train_subset = 10000

graph = tf.Graph()
with graph.as_default():
    # Input data.
   tf_train_dataset = tf.constant(train_dataset[:train_subset, :])
   tf_train_labels = tf.constant(train_labels[:train_subset])
   tf_valid_dataset = tf.constant(valid_dataset)
   tf_test_dataset = tf.constant(test_dataset)

    # Variables.
    # These are the parameters that we are going to be training. The weight
    # matrix will be initialized using random values following a (truncated)
    # normal distribution. The biases get initialized to zero.
   hidden_layer_size = 1024
   hidden_weights = tf.Variable(
           tf.truncated_normal([image_size * image_size, hidden_layer_size]))
   hidden_biases = tf.Variable(tf.zeros([hidden_layer_size]))
   hidden_layer = tf.nn.relu(tf.matmul(tf_train_dataset, hidden_weights) + hidden_biases)
   output_weights = tf.Variable(
       tf.truncated_normal([hidden_layer_size, num_labels]))
   output_biases = tf.Variable(tf.zeros([num_labels]))

   logits = tf.matmul(hidden_layer, output_weights) + output_biases
   loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
   l2_regularizer = tf.nn.l2_loss(output_weights) + tf.nn.l2_loss(hidden_weights)
   loss += 1e-4 * l2_regularizer
    # Optimizer.
   optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

    # Predictions for the training, validation, and test data.
   train_prediction = tf.nn.softmax(logits)
    

  # Setup validation prediction step.
   valid_hidden = tf.nn.relu(tf.matmul(tf_valid_dataset, hidden_weights) + hidden_biases)
   valid_logits = tf.matmul(valid_hidden, output_weights) + output_biases
   valid_prediction = tf.nn.softmax(valid_logits)

  # And setup the test prediction step.
   test_hidden = tf.nn.relu(tf.matmul(tf_test_dataset, hidden_weights) + hidden_biases)
   test_logits = tf.matmul(test_hidden, output_weights) + output_biases
   test_prediction = tf.nn.softmax(test_logits)
num_steps = 801

def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

with tf.Session(graph=graph) as session:
  
  tf.global_variables_initializer().run()
  for step in range(num_steps):
    
    _, l, predictions = session.run([optimizer, loss, train_prediction])
    if (step % 100 == 0):
      print('Loss at step %d: %f' % (step, l))
      print('Training accuracy: %.1f%%' % accuracy(
        predictions, train_labels[:train_subset, :]))
      
      print('Validation accuracy: %.1f%%' % accuracy(
        valid_prediction.eval(), valid_labels))
  print('Test accuracy: %.1f%%' % accuracy(test_prediction.eval(), test_labels))
