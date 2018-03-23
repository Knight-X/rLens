#!/usr/bin/env python3

# Copyright 2016 Google Inc.
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

# TODO:
# - Implement approximate value functions

import argparse
import collections
import curses
import random
import sys
import time
import socket
import os


import context
import policy_gradient
import tensorflow as tf
import environment as en

HOST = '127.0.0.1'
PORT = 1992

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
class Player(object):
  '''A Player provides input to the game as a simulation evolves.'''
  def interact(self, ctx, sim):
    # All players have the same interface
    # pylint: disable=unused-argument
    pass

class Mission(object):
    def __init__(self, ctx, env, driver):
        self._context = ctx
        self._env = env
        self._driver = driver
        self._was_in_terminal_state = False

    def step(self):
        self._driver.interact(self._context, self._env)
        self._was_in_terminal_state = self._env.in_terminal_state

def gen(actionset):
    idx2regs = [a for a in actionset]
    idx2regs.sort()
    regs2idx = {}
    for i in range(len(idx2regs)):
        regs2idx[idx2regs[i]] = i
    return idx2regs, regs2idx
def main():
  parser = argparse.ArgumentParser(description='Simple Reinforcement Learning.')
  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument('--interactive', action='store_true',
                     help='use the keyboard arrow keys to play')
  group.add_argument('--pg', action='store_true',
                     help='play automatically with policy gradients')
  parser.add_argument('--random', action='store_true',
                      help='generate a random map')

  args = parser.parse_args()

  ctx = context.Context()
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind((HOST, PORT))
  env = en.Environment()
  rplayer = policy_gradient.RandomPlayer(sock)
  actionset = rplayer.interact(env)
  idx2regs, regs2idx = gen(actionset)

  if True:
    g = tf.Graph()
    s = tf.Session(graph=g)
    player = policy_gradient.PolicyGradientPlayer(g, s, [247, 247], sock, idx2regs, regs2idx)
    with g.as_default():
      init = tf.global_variables_initializer()
      s.run(init)
  else:
    sys.exit(1)

  go = Mission(ctx, env, player)
  ctx.run_loop.post_task(go.step, repeat=True)
  ctx.start()


if __name__ == '__main__':
  main()
