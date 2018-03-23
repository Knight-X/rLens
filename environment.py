import numpy as np

import subprocess
import os


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


