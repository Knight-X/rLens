from subprocess import call
import cv2
import numpy as np
from skimage.transform import resize

def outputdict(reward):
  actions = {}
  g = open("beforeparsecandidate.txt", "w")
  for i in range(len(reward) / 2):
    g.write(str(reward[i]))
    g.write(' ')
  g.close()
  for i in range(len(reward) / 2):
      actions[reward[i * 2]] = reward[i * 2 + 1]
  g = open("parsecandidate.txt", "w")
  for i in actions.iteritems():
    g.write(str(i))
    g.write(' ')
  g.close()
  return actions

def fileToImage(state):
    infile = open(state, "r").readlines()
    arr = np.zeros((247, 247))
    img = np.copy(arr)
    minimum = 999999
    rewardline = 0
    for index in range(0, len(infile)):
        line = infile[index].split("&")
        if line[0] == "reward\n":
            rewardline = index+1
            rewarddata = infile[rewardline].split("&") 
            reward_dic = outputdict(rewarddata)
            return img, reward_dic
        print "firstline: " + line[0]
        for x in range(1, len(line), 2):
          if int(line[x]) > len(arr) or int(line[x + 1]) > len(arr):
            arr.resize((int(line[x + 1]) + 1, 247))
          if int(line[x]) < minimum:
            minimum = int(line[x])
          print "begin: " + line[x] + "end: " + line[x+1]
          for r in range(int(line[x]), int(line[x + 1]) + 1):
            arr[r][int(line[0])] = 255
    res = np.split(arr, arr.shape[0])
    img = np.copy(res[minimum])
    black = np.zeros((1, 247))
    if minimum + 247 > len(res):
      for tmp in range(len(res), minimum + 247):
        res.append(black)
    for sub in range(minimum + 1, minimum + 247):
      img = np.concatenate((img, res[sub]), axis=0)
    shape = img.shape
    if shape[0] != 247 and shape[1] != 247:
        print "dim wrong"
    rewarddata = infile[rewardline].split("&") 
    reward_dic = outputdict(rewarddata)
    cv2.imwrite("filename.png", img)
    return img, reward_dic


""" for i in range(100, 101):
  name = "go" + str(i) + ".txt"
  print name
  infile = open(name, "r").readlines()
  arr = np.zeros((247, 247))
  minimum = 999999 
  for index in range(0, len(infile)):
    line = infile[index].split("&")
    print "firstline: " + line[0]
    for x in range(1, len(line), 2):
      if int(line[x]) > len(arr) or int(line[x + 1]) > len(arr):
          arr.resize((int(line[x + 1]) + 1, 247))
      if int(line[x]) < minimum:
          minimum = int(line[x])
      print "begin: " + line[x] + "end: " + line[x+1]
      for r in range(int(line[x]), int(line[x + 1]) + 1):
          arr[r][int(line[0])] = 255
  res = np.split(arr, arr.shape[0])
  img = res[minimum]
  black = np.zeros((1, 247))
  if minimum + 247 > len(res):
    for tmp in range(len(res), minimum + 247):
      res.append(black)
  for sub in range(minimum + 1, minimum + 247):
      img = np.concatenate((img, res[sub]), axis=0)
  
  cv2.imwrite("filename.png", img)"""
