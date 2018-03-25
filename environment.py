import numpy as np

import subprocess
import random
import os
import struct
import parse


class Environment(object):

  def act(self, action):
        f = open("action.txt", "w")
        f.write(str(action))
        f.close()
        g = open("actionbarrier.txt", "w")
        g.close()
  def in_terminal_state(self):
        if os.path.isfile("terminal.txt") == True:
            os.remove("terminal.txt")
            return True
        return False
  def reset(self, process):
      process.terminate()
      process.wait()
      print "process finish"
      process = subprocess.Popen(['../llvm-reg/llvm/build/bin/llc', '-debug-only=regallocdl', '--regalloc=drl', 'add.ll', '-o', 'convba.s'],shell=False, stdout=subprocess.PIPE)
      return process
  
  def terminate(self, process):
      process.terminate()
      process.wait()
      print "process finish"

class RandomPlayer:
  def __init__(self, sock):
    self._iter = 1
    self._p = subprocess.Popen(['../llvm-reg/llvm/build/bin/llc', '-debug-only=regallocdl', '--regalloc=drl', 'add.ll', '-o', 'convba.s'],shell=False, stdout=subprocess.PIPE)
    self._sock = sock
    self._sock.listen(5)

  def interact(self):
    terminal = False
    actions = set()
    while (terminal == False):
      print "start accept " + str(self._iter)
      conn, addr = self._sock.accept()
      data = conn.recv(1024)
      if (data[0] == 'e'):
        terminal = True
        break
      reward_map = self.getState(data)
      action = self.doAction(reward_map)
      conn.send(str(action))
      self._iter = self._iter + 1
      for g in reward_map.keys():
        actions.add(g)

    self._iter = 1 
    self._totalr = 0.0
    self.terminate()
    return actions 

  def getState(self, data):

      data = struct.unpack("!i", data)[0]
      if int(data) != self._iter:
          print "c++ iter: " + data + " python iter: " + str(self._iter)
          sys.exit(0)
      state, reward_map, _ = parse.fileToImage("state.txt", self._iter)
      return reward_map

  def doAction(self, reward_map):
      action = reward_map.keys()[0]
      return action 
  def terminate(self):
    self._p.terminate()
    self._p.wait()
    print "process finish"

class Gplayer:
  def __init__(self, sock, idx2regs, regs2idx):
    self._sock = sock
    self._sock.listen(5)
    self._iter = 1
    self._totalr = 0.0
    self._idx2Regs = idx2regs
    self._regs2idx = regs2idx
    self.best_reward = 0
    self.process_init = False

  def terprocess(self):
      print "terminal the process in python"
      self._p.terminate()
      self._p.wait()

  def reset(self):
    self._p = subprocess.Popen(['../llvm-reg/llvm/build/bin/llc', '-debug-only=regallocdl', '--regalloc=drl', 'add.ll', '-o', 'convba.s'],shell=False, stdout=subprocess.PIPE)
    print "start accept " + str(self._iter)
    self._iter = 1
    self._conn, addr = self._sock.accept()
    data = self._conn.recv(1024)
    state, reward_map = self.getState(data)
    return state, reward_map

  def step(self, action):
    reward = 0.0
    self._conn.send(str(action))
    self._iter = self._iter + 1
    self._conn, addr = self._sock.accept()
    data = self._conn.recv(1024)
    if data[0] == 'e':
        self.terprocess()
        return [], True, []
    state, reward_map = self.getState(data)
    return state, False, reward_map

  def getState(self, data):
    data = struct.unpack("!i", data)[0]
    if int(data) != self._iter:
      print "c++ iter: " + data + " python iter: " + str(self._iter)
      sys.exit(0)
    state, reward_map, _ = parse.fileToImage("state.txt", self._iter)
    return state, reward_map

  def among(self, distri, reward_map, ac, valid):
    if valid and reward_map.get(str(ac)) != None:
        reward = reward_map[str(ac)]
        return int(reward), ac, True
    elif valid and reward_map.get(str(ac)) == None:
        return -10, ac, False
    elif not valid and reward_map.get(str(ac)) != None:
        reward = reward_map[str(ac)]
        return int(reward), ac, True
    finalidx2reg = {}
    index = 0
    actions = [] 
    for i in self._regs2idx.keys():
        if reward_map.get(str(i)) != None:
            actions.insert(index, distri[self._regs2idx[str(i)]])
            finalidx2reg[str(index)] = i
            index = index + 1   
    if random.random() < 0.05:
      action = np.random.choice(index, 1, actions)
      action = action[0]
    else:
      action = np.argmax(np.array(actions))
    action = finalidx2reg[str(action)]
    reward = reward_map[str(action)]
    return int(reward), action, True

def among(distri, ac, valid):
    reward_map = {"1": 3, "0": 5}
    if valid and reward_map.get(str(ac)) != None:
        return ac, True
    elif valid and reward_map.get(str(ac)) == None:
        return ac, False
    elif not valid and reward_map.get(str(ac)) != None:
        return ac, True
    
    index = 2 
    actions = []
    for i in range(index):
        actions.append(distri[0][i])
    #if random.random() < 0.0005:
    #  action = np.random.choice(index, 1, actions)
    #  action = action[0]
    #else:
    action = np.argmax(np.array(actions))
    return action, True
