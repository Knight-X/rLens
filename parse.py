from subprocess import call
import cv2
import numpy as np
from skimage.transform import resize

def outputdict(reward):
  actions = {}
  for i in range(len(reward) / 2):
      actions[reward[i * 2]] = reward[i * 2 + 1]
  return actions

def fileToImage(state, iteration):
    infile = open(state, "r").readlines()
    vreg =infile[0].split("&")
    slotstart = int(vreg[1])
    slotend = int(vreg[2])
    minimum = slotstart 
    if slotstart >= 10:
        minimum = slotstart - 10
    else:
        minimum = 0
    arr = np.zeros((max(slotend + 1, 247), 247))
    img = np.copy(arr)

    rewarddata = infile[2].split("&")
    reward_dic = outputdict(rewarddata)

    vrewarddata = infile[4].split("&")
    vreward_dic = outputdict(vrewarddata)
    for index in range(5, len(infile)):
        line = infile[index].split("&")
        if line[0] == "reward\n" or line[0] == "3333":
            continue;

        #print "firstline: " + line[0]
        for x in range(1, len(line), 2):
          if int(line[x]) > len(arr) or int(line[x + 1]) > len(arr):
            arr.resize((int(line[x + 1]) + 1, 247))
          if slotstart > len(arr) or slotend > len(arr):
            arr.resize(slotend + 1, 247)
          #print "begin: " + line[x] + "end: " + line[x+1]
          for r in range(int(line[x]), int(line[x + 1]) + 1):
            arr[r][int(line[0])] = 1
    for key in reward_dic:
        for sloti in range(slotstart, min(slotstart + 247, slotend)):
            arr[sloti][int(key)] = 2 
    for key in vreward_dic:
        for sloti in range(slotstart, min(slotstart + 247, slotend)):
            arr[sloti][int(key)] = 3 
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
    name = "filename" + str(iteration) + ".png"
    cv2.imwrite(name, img)
    reward_dic.update(vreward_dic)
    return img, reward_dic, arr


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
