# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Emanuele Ruffaldi 2017
# ==============================================================================

"""A deep MNIST classifier using convolutional layers.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/pros
"""
# Disable linter warnings to maintain consistency with tutorial.
# pylint: disable=invalid-name
# pylint: disable=g-bad-import-order

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import math
import argparse
import sys
import json
import numpy as np
import uuid
import tensorflow.compat.v1 as tf
import tensorflow_datasets
from tensorflow.python.client import timeline
import tensorflow.compat.v1 as tf

import tensorflow as tf

from sys import platform
FLAGS = None

def machine():
  return dict(linux="glx64",darwin="maci64",win32="win32").get(platform)


def getAccuracy(matrix):
  #sum(diag(mat))/(sum(mat))
  sumd = np.sum(np.diagonal(matrix))
  sumall = np.sum(matrix)
  sumall = np.add(sumall,0.00000001)
  return sumd/sumall

def getPrecision(matrix):
  #diag(mat) / rowSum(mat)
  sumrow = np.sum(matrix,axis=1)
  sumrow = np.add(sumrow,0.00000001)
  precision = np.divide(np.diagonal(matrix),sumrow)
  return np.sum(precision)/precision.shape[0]

def getRecall(matrix):
  #diag(mat) / colsum(mat)
  sumcol = np.sum(matrix,axis=0)
  sumcol = np.add(sumcol,0.00000001)
  recall = np.divide(np.diagonal(matrix),sumcol)
  return np.sum(recall)/recall.shape[0]

def getSensitivity(matrix):
  return 0;

def getSpecificity(matrix):
  return 0;

def get2f(matrix):
  #2*precision*recall/(precision+recall)
  precision = getPrecision(matrix)
  recall = getRecall(matrix)
  return (2*precision*recall)/(precision+recall)


def deepnn(x,filter1_size=5,features1=32,filter2_size=5,features2=64,densesize=1024,classes=10):
  """deepnn builds the graph for a deep net for classifying digits.

  Args:
    x: an input tensor with the dimensions (N_examples, 784), where 784 is the
    number of pixels in a standard MNIST image.

  Returns:
    A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
    equal to the logits of classifying the digit into one of 10 classes (the
    digits 0-9). keep_prob is a scalar placeholder for the probability of
    dropout.
  """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([filter1_size, filter1_size, 1, features1])
  b_conv1 = bias_variable([features1])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([filter2_size, filter2_size, features1, features2])
  b_conv2 = bias_variable([features2])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.

  # 7 is imagewidth/4
  W_fc1 = weight_variable([7 * 7 * features2, densesize])
  b_fc1 = bias_variable([densesize])
    
  h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*features2])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)
  h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([densesize, classes])
  b_fc2 = bias_variable([classes])

  y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
  return y_conv, keep_prob


def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  dsTR = tensorflow_datasets.load('mnist',split='train')
  dsTR = ds.shuffle(1024).batch(FLAGS.batchsize).prefetch(tf.data.AUTOTUNE)
  dsTE = tensorflow_datasets.load('mnist',split='test')

  #input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  tf.compat.v1.disable_eager_execution()


  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # Build the graph for the deep net
  y_conv, keep_prob = deepnn(x,filter1_size=FLAGS.filter1,filter2_size=FLAGS.filter2,features1=FLAGS.features2,features2=FLAGS.features2,densesize=FLAGS.dense)

  total_parameters = 0
  for variable in tf.trainable_variables():
      # shape is an array of tf.Dimension
      shape = variable.get_shape()
      print(shape)
      print(len(shape))
      variable_parametes = 1
      for dim in shape:
          print(dim)
          variable_parametes *= dim.value
      print(variable_parametes)
      total_parameters += variable_parametes
  print("Total Parameters",total_parameters)


  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

  if FLAGS.adam:
    train_step = tf.train.AdamOptimizer(FLAGS.adam_rate).minimize(cross_entropy) #GradientDescentOptimizer(0.5).minimize(cross_entropy)
  else:
    train_step = tf.train.GradientDescentOptimizer(FLAGS.gradient_rate).minimize(cross_entropy) #GradientDescentOptimizer(0.5).minimize(cross_entropy)

  kw = {}
  if FLAGS.no_gpu:
    kw["device_count"] = {'GPU': 0  }
  if FLAGS.singlecore:
    kw["intra_op_parallelism_threads"]=1
    kw["inter_op_parallelism_threads"]=1

  iterations = FLAGS.epochs*int(math.ceil(60000.0/FLAGS.batchsize))
  config = tf.ConfigProto(**kw)
  sess = tf.InteractiveSession(config=config)
  #train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  predictions = tf.argmax(y_conv, 1)
  correct_prediction = tf.equal(predictions, tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  run_metadata = tf.RunMetadata()
  if FLAGS.trace:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
  else:
    run_options = tf.RunOptions()

  if True: #with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t0 = time.time()
    for i in range(iterations):
      batch =next(dsTR)
      if False and i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch[0], y_: batch[1], keep_prob: 1.0}, options=run_options, run_metadata=run_metadata)
        print('step %d, training accuracy %g' % (i, train_accuracy))
      _,cross_entropy_value = sess.run([train_step,cross_entropy], feed_dict={x: batch_xs, y_: batch_ys,keep_prob: 0.5})
      losses[i] = cross_entropy_value
      #train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    training_time = time.time()-t0
    print ("training_time",training_time)
    print ("iterations",iterations)
    print ("batchsize",FLAGS.batchsize)

    t0 = time.time()
    evaliterations = int(math.ceil(10000.0/FLAGS.batchsize))
    accuracyvalue = 0
    accuracyvaluecount = 0
    cm = None
    for i in range(evaliterations):
      batch = next(dsTE)
      accuracyvalue += sess.run(accuracy,feed_dict={
        x: batch[0], y_: batch[1], keep_prob: 1.0},)
      accuracyvaluecount += 1
      cma = sess.run(tf.contrib.metrics.confusion_matrix(tf.argmax(y_, 1),predictions,10),feed_dict={x: batch[0],y_: batch[1], keep_prob: 1.0})
      if cm is None:
        cm =  cma
      else:
        cm += cma
    accuracyvalue /= accuracyvaluecount

    test_time = time.time()-t0
    print('test accuracy %g' % accuracyvalue)

    print (cm)
    cm_accuracy = getAccuracy(cm)
    cm_Fscore = get2f(cm)
    cm_sensitivity = getSensitivity(cm)
    cm_specificity = getSpecificity(cm)
    print ("test CM accuracy",cm_accuracy,"CM F1",cm_Fscore)

    if FLAGS.trace:
      # Create the Timeline object, and write it to a json
      tl = timeline.Timeline(run_metadata.step_stats)
      ctf = tl.generate_chrome_trace_format(show_memory=True)
      with open('cnn_accuracy_timeline.json', 'w') as f:
          f.write(ctf)
      graph_def = tf.get_default_graph().as_graph_def()
      json_string = json_format.MessageToJson(graph_def)
      with open('cnn_structure.json', 'w') as f:
        f.write(json_string)

    go = str(uuid.uuid1())+'.json';
    args = FLAGS
    out = dict(accuracy=float(accuracyvalue),training_time=training_time,single_core=1 if args.singlecore else 0,implementation="tf",type='single',test='cnn',gpu=0 if args.no_gpu else 1,machine=machine(),epochs=args.epochs,batchsize=args.batchsize,now_unix=time.time(),cnn_specs=(args.filter1,args.filter2,args.features1,args.features2,args.dense),cm_accuracy=float(cm_accuracy),cm_Fscore=float(cm_Fscore),iterations=iterations,testing_time=test_time,total_params=total_parameters,cm_specificity=float(cm_specificity),cm_sensitivity=float(cm_sensitivity))
    open(go,"w").write(json.dumps(out))
    np.savetxt(go+".loss.txt", losses)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('-a', '--filter1',type=int,default=5,help='first filter size');
  parser.add_argument('-b','--filter2',type=int,default=5,help='second filter size');
  parser.add_argument('-d','--dense',type=int,default=1024,help='dense bank');
  parser.add_argument('-A','--features1',type=int,default=32,help='features of first');
  parser.add_argument('-B','--features2',type=int,default=64,help='features of second');
  parser.add_argument('--original',action="store_true",help='picks original Tensorflow values (3.5M parameters)')
  parser.add_argument('--light',action="store_true",help='light values (400k parameters)')
  parser.add_argument('--lighter',action="store_true",help='lighter values (100k parameters)')
  parser.add_argument('--p57',action="store_true",help='(57k parameters)')

  parser.add_argument('--no-gpu',action="store_true")
  parser.add_argument('--trace',action="store_true")
  parser.add_argument('--singlecore',action="store_true")
  parser.add_argument('--adam',action="store_true")
  parser.add_argument('--adam_rate',default=1e-4,type=float)
  parser.add_argument('--gradient_rate',default=0.5,type=float)
  parser.add_argument('--epochs',help="epochs",default=10,type=int)
  parser.add_argument('--batchsize',help="batch size",type=int,default=100)
  parser.add_argument('-w',action="store_true")
  FLAGS, unparsed = parser.parse_known_args()
  if FLAGS.original:
    # 1M
    FLAGS.filter1 = 5
    FLAGS.filter2 = 5
    FLAGS.dense = 1024
    FLAGS.features1 = 32
    FLAGS.features2 = 64
  elif FLAGS.light:
    # 400k
    FLAGS.filter1 = 5
    FLAGS.filter2 = 5
    FLAGS.dense = 256
    FLAGS.features1 = 16
    FLAGS.features2 = 32
  elif FLAGS.lighter:
    # 100k
    FLAGS.filter1 = 5
    FLAGS.filter2 = 5
    FLAGS.dense = 128
    FLAGS.features1 = 16
    FLAGS.features2= 16
  elif FLAGS.p57:
    FLAGS.filter1 = 5
    FLAGS.filter2 = 5
    FLAGS.dense = 64
    FLAGS.features1 = 16
    FLAGS.features2= 16 
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
