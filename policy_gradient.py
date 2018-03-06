# Copyright 2017 Google Inc.
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

'''A policy gradient-based maze runner in TensorFlow.

This is based on
https://medium.com/@awjuliani/super-simple-reinforcement-learning-tutorial-part-2-ded33892c724
'''

import collections
import subprocess
import operator
import time
import os
from shutil import copyfile
import numpy as np
import sys
import tensorflow as tf
import sys
import socket

import grid
import parse


_EMBEDDING_SIZE = 3
_FEATURE_SIZE = 3
_ACTION_SIZE = 255


class PolicyGradientNetwork(object):
  '''A policy gradient network.

  This has a number of properties for operating on the network with
  TensorFlow:

    state: Feed a world-sized array for selecting an action.
        [-1, h, w] int32
    action_out: Produces an action, fed state.

    action_in: Feed the responsible actions for the rewards.
        [-1, 1] int32 0 <= x < len(movement.ALL_ACTIONS)
    advantage: Feed the "goodness" of each given state.
        [-1, 1] float32
    update: Given batches of experience, train the network.

    loss: Training loss.
    summary: Merged summaries.
  '''

  def __init__(self, name, graph, world_size_h_w):
    '''Creates a PolicyGradientNetwork.

    Args:
      name: The name of this network. TF variables are in a scope with
          this name.
      graph: The TF graph to build operations in.
      world_size_h_w: The size of the world, height by width.
    '''
    h, w = world_size_h_w
    self._h, self._w = h, w
    with graph.as_default():
      initializer = tf.contrib.layers.xavier_initializer()
      with tf.variable_scope(name) as self.variables:
        self.state = tf.placeholder(tf.int32, shape=[None, h, w])

        # Input embedding
        embedding = tf.get_variable(
            'embedding', shape=[_FEATURE_SIZE, _EMBEDDING_SIZE],
            initializer=initializer)
        embedding_lookup = tf.nn.embedding_lookup(
            embedding, tf.reshape(self.state, [-1, h * w]),
            name='embedding_lookup')
        embedding_lookup = tf.reshape(embedding_lookup,
                                      [-1, h, w, _EMBEDDING_SIZE])

        # First convolution.
        conv_1_out_channels = 27
        conv_1 = tf.contrib.layers.conv2d(
            trainable=True,
            inputs=embedding_lookup,
            num_outputs=conv_1_out_channels,
            kernel_size=[3, 3],
            stride=1,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            # TODO: What's a good initializer for biases? Below too.
            biases_initializer=initializer)

        shrunk_h = h
        shrunk_w = w

        # Second convolution.
        conv_2_out_channels = 50
        conv_2_stride = 2
        conv_2 = tf.contrib.layers.conv2d(
            trainable=True,
            inputs=conv_1,
            num_outputs=conv_2_out_channels,
            kernel_size=[5, 5],
            stride=conv_2_stride,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)
        shrunk_h = (h + conv_2_stride - 1) // conv_2_stride
        shrunk_w = (w + conv_2_stride - 1) // conv_2_stride

        # Third convolution.
        conv_3_out_channels = 100
        conv_3_stride = 2
        conv_3 = tf.contrib.layers.conv2d(
            trainable=True,
            inputs=conv_2,
            num_outputs=conv_3_out_channels,
            kernel_size=[5, 5],
            stride=conv_3_stride,
            padding='SAME',
            activation_fn=tf.nn.relu,
            weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)
        shrunk_h = (shrunk_h + conv_3_stride - 1) // conv_3_stride
        shrunk_w = (shrunk_w + conv_3_stride - 1) // conv_3_stride

        # Resupply the input at this point.
        resupply = tf.concat([
              tf.reshape(conv_3,
                         [-1, shrunk_h * shrunk_w * conv_3_out_channels]),
              tf.reshape(embedding_lookup, [-1, h * w * _EMBEDDING_SIZE])
            ], 1, name='resupply')

        # First fully connected layer.
        connected_1 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=resupply,
            num_outputs=h+w,
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer
            )

        # Second fully connected layer, steps down.
        connected_2 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=connected_1,
            num_outputs=17,
            activation_fn=tf.nn.relu,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer 
            )

        # Logits, softmax, random sample.
        connected_3 = tf.contrib.layers.fully_connected(
            trainable=True,
            inputs=connected_2,
            num_outputs=_ACTION_SIZE,
            activation_fn=tf.nn.sigmoid,
            weights_initializer=initializer,
            weights_regularizer=tf.contrib.layers.l2_regularizer(1.0),
            biases_initializer=initializer)
        self.action_softmax = tf.nn.softmax(connected_3, name='action_softmax')

        # Sum the components of the softmax
#        probability_histogram = tf.cumsum(self.action_softmax, axis=1)
#        sample = tf.random_uniform(tf.shape(probability_histogram)[:-1])
#        filtered = tf.where(probability_histogram >= sample,
#                            probability_histogram,
#                            tf.ones_like(probability_histogram))

        
        #self.filtered = filtered
        self.action_out = tf.argmin(self.action_softmax, 1)

        self.action_in = tf.placeholder(tf.int32, shape=[None, 1])
        self.advantage = tf.placeholder(tf.float32, shape=[None, 1])

        action_one_hot = tf.one_hot(self.action_in, _ACTION_SIZE,
                                    dtype=tf.float32)
        action_advantage = self.advantage * action_one_hot
        loss_policy = -tf.reduce_mean(
            tf.reduce_sum(tf.log(self.action_softmax) * action_advantage, 1),
            name='loss_policy')
        # TODO: Investigate whether regularization losses are sums or
        # means and consider removing the division.
        #loss_regularization = (0.05 / tf.to_float(tf.shape(self.state)[0]) *
        #    sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)))
        #self.loss = loss_policy + loss_regularization
        self.loss = loss_policy

        #tf.summary.scalar('loss_policy', loss_policy)
        #tf.summary.scalar('loss_regularization', loss_regularization)

        # TODO: Use a decaying learning rate
        #optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        #initial_learning_rate = 0.1
        #global_step = tf.Variable(0, trainable=False)

        #learning_rate = tf.train.exponential_decay(initial_learning_rate,
        #                                           global_step=global_step,
        #                                           decay_steps=100000,decay_rate=0.96)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0000625)
        self.update = optimizer.minimize(self.loss)

        #self.summary = tf.summary.merge_all()

  def predict(self, session, states):
    '''Chooses actions for a list of states.

    Args:
      session: The TensorFlow session to run the net in.
      states: A list of simulation states which have been serialized
          to arrays.

    Returns:
      An array of actions, 0 .. 4 and an array of array of
      probabilities.
    '''
    return session.run([self.action_out, self.action_softmax],
                       feed_dict={self.state: states})

  def train(self, session, episodes):
    '''Trains the network.

    Args:
      episodes: A list of episodes. Each episode is a list of
          3-tuples with the state, the chosen action, and the
          reward.
    '''
    size = sum(map(len, episodes))
    state = np.empty([size, self._h, self._w])
    action_in = np.empty([size, 1])
    advantage = np.empty([size, 1])
    i = 0
    print "start traing... batch_size: " + str(size)
    for episode in episodes:
      episode_size = len(episode)
      r = 0.0
      for step_state, action, reward in reversed(episode):
        state[i,:,:] = step_state
        action_in[i,0] = action
        r = (reward / episode_size)+ 0.97 * r
        advantage[i,0] = r
        i += 1
    # Scale rewards to have zero mean, unit variance
    advantage = (advantage - np.mean(advantage)) / np.var(advantage)

    return session.run([self.loss, self.update], feed_dict={
        self.state: state,
        self.action_in: action_in,
        self.advantage: advantage
      })


_EXPERIENCE_BUFFER_SIZE = 30


class PolicyGradientPlayer(grid.Player):
  def __init__(self, graph, session, world_size_w_h, sock):
    super(PolicyGradientPlayer, self).__init__()
    w, h = world_size_w_h
    self._net = PolicyGradientNetwork('net', graph, (h, w))
    self._experiences = collections.deque([], _EXPERIENCE_BUFFER_SIZE)
    self._experience = []
    self._session = session
    self._p = subprocess.Popen(['./llvm-reg/llvm/build/bin/llc', '-debug-only=regallocdl', '--regalloc=drl', 'foo.ll', '-o', 'convba.s'],shell=False, stdout=subprocess.PIPE)
    self._iter = 1
    self._sock = sock
    self._sock.listen(5)
    self._totalr = 0.0
    print "start listen"

  def interact(self, ctx, env):
    print "start accept " + str(self._iter)
    conn, addr = self._sock.accept()
    data = conn.recv(1024)
    #terminal = env.in_terminal_state()
    if data[0] == 'e':
        terminal = True
    else:
        terminal = False
    if terminal:
      self._experiences.append(self._experience)
      self._experience = []
      summary, _ = self._net.train(self._session, self._experiences)
      self._p = env.reset(self._p)
      self._iter = 1 
      print "loss"
      print summary
      with open("result3.txt", "a") as myfile:
          con = "loss: " + str(summary) + " score: " + str(self._totalr)
          myfile.write(con)
          myfile.write("\n")
      self._totalr = 0.0
    else:
      state, reward_map = self.getState(data)
      reward, action = self.doAction(state, reward_map, env)
      conn.send(str(action))
      self._experience.append((state, action, float(reward)))
      self._iter = self._iter + 1
      self._totalr = self._totalr + float(reward)
  def getState(self, data):

      if int(data) == self._iter:
          print data
          sys.exit(0)
      state, reward_map, _ = parse.fileToImage("state.txt", self._iter)
      return state, reward_map

  def choose(self, reward, filtered):
      print reward
      actions = {} 
      for i in range(255):
          if reward.get(str(i)) == None:
              filtered[i] = -1.0
          else:
              actions[i] = filtered[i]
      print "actions result:"
      print actions

      return np.argmax(filtered)

      

  def doAction(self, state, reward_map, env):
      res_action = 0
      reward = 0.0
      [[action], [softmax]] = self._net.predict(self._session, [state])
      action = self.choose(reward_map, softmax)
      env.act(action)
      res_action = action
      reward = reward_map[str(action)]
      return reward, res_action 
