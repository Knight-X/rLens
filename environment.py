import numpy as np

class Environment(object):

  def act(self, action):
        f = open("action.txt", "w")
        f.write(str(action))
        f.close()
        g = open("actionbarrier.txt", "w")
        g.close()
  def in_terminal_state(self):
        return False
        if os.path.isfile("terminal.txt") == True:
            os.remove("terminal.txt")
            return True

