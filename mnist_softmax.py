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
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
https://www.tensorflow.org/get_started/mnist/beginners
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  # Import data
  print ("updated")
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.matmul(x, W) + b

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10])

  # The raw formulation of cross-entropy,
  #
  #   tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(y)),
  #                                 reduction_indices=[1]))
  #
  # can be numerically unstable.
  #
  # So here we use tf.nn.softmax_cross_entropy_with_logits on the raw
  # outputs of 'y', and then average across the batch.
  cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
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

  config = tf.ConfigProto(**kw)
  sess = tf.InteractiveSession(config=config)
  tf.global_variables_initializer().run()
  # Train
  t0 = time.time()
  for _ in range(FLAGS.iter):
    batch_xs, batch_ys = mnist.train.next_batch(FLAGS.batch)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  print ("training_time",time.time()-t0)
  print ("iterations",FLAGS.iter)
  print ("batchsize",FLAGS.batch)
  # Test trained model
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels}))

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  parser.add_argument('--no-gpu',action="store_true")
  parser.add_argument('--singlecore',action="store_true")
  parser.add_argument('--adam',action="store_true")
  parser.add_argument('--adam_rate',default=1e-4,type=float)
  parser.add_argument('--gradient_rate',default=0.5,type=float)
  parser.add_argument('--iter',help="iterations",default=6000)
  parser.add_argument('--batch',help="batch size",default=100)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
