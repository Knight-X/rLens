from subprocess import call
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import numpy as np
from skimage.transform import resize
from shutil import copy2
import os
import sys

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
    maxmum = slotend
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
    #for index in range(5, len(infile)):
    #    line = infile[index].split("&")
    #    if line[0] == "reward\n" or line[0] == "3333":
    #        continue;

        #print "firstline: " + line[0]
    #    for x in range(1, len(line), 2):
    #      if int(line[x]) > len(arr) or int(line[x + 1]) > len(arr):
    #        arr.resize((int(line[x + 1]) + 1, 247))
    #      if slotstart > len(arr) or slotend > len(arr):
    #        arr.resize(slotend + 1, 247)
          #print "begin: " + line[x] + "end: " + line[x+1]
    #      for r in range(int(line[x]), int(line[x + 1]) + 1):
    #        arr[r][int(line[0])] = 255 
    #for key in reward_dic:
    #    for sloti in range(slotstart, min(slotstart + 247, slotend)):
    #        arr[sloti][int(key)] = 200 
    #for key in vreward_dic:
    #    for sloti in range(slotstart, min(slotstart + 247, slotend)):
    #        arr[sloti][int(key)] = 100 
    #res = np.split(arr, arr.shape[0])
    #img = np.copy(res[minimum])
    #black = np.zeros((1, 247))
    #if minimum + 247 > len(res):
    #  for tmp in range(len(res), minimum + 247):
    #    res.append(black)
    #for sub in range(minimum + 1, minimum + 247):
    #  img = np.concatenate((img, res[sub]), axis=0)
    #shape = img.shape
    #if shape[0] != 247 and shape[1] != 247:
    #    print "dim wrong"
    #name = "filename" + str(iteration) + ".png"
    #cv2.imwrite(name, img)
    reward_dic.update(vreward_dic)
    img = img.flatten()
    return img, reward_dic, maxmum, arr

def physicalre(a, reward_dic, ratio, actionsize, slotstart, slotend, g):
    for key in reward_dic:
        for j in range(actionsize):
            if j * ratio <= slotstart < (j + 1) * ratio or slotstart <= j * ratio <= slotend or j * ratio <= slotend < (j + 1) * ratio:
                    a[j][g[str(key)]] = 255
                    
def vrreward(a, vreward_dic, ratio, actionsize, slotstart, slotend, g):
    for key in vreward_dic:
        for j in range(actionsize):
            if j * ratio <= slotstart < (j + 1) * ratio or slotstart <= j * ratio <= slotend or j * ratio <= slotend < (j + 1) * ratio:
                if a[j][g[str(key)]] == 255:
                    print "wrong virtual regsiter"
                    sys.exit(0)
                else:
                    a[j][g[str(key)]] = 75

def getstate(state, iteration, maxlength, actionsize, reg2idx, tofile):
    directory = "./data/log"
    infile = open(state, "r").readlines()                  
    vreg = infile[0].split("&")
    slotstart = int(vreg[1])
    slotend = int(vreg[2]) 
    if maxlength % actionsize == 0:
        ratio = maxlength / actionsize
    else:                  
        ratio = (maxlength / actionsize) + 1
    rewarddata = infile[2].split("&")
    reward_dic = outputdict(rewarddata)
    vrewarddata = infile[4].split("&")
    vreward_dic = outputdict(vrewarddata)
    a = np.zeros((actionsize, actionsize))
                           
    for index in range(5, len(infile)):
        line = infile[index].split("&")
        if line[0] == "reward\n" or line[0] == "3333":
            continue;            
        reg = line[0]
        if reg2idx.get(str(reg)) == None:
            continue
        for x in range(1, len(line), 2):
            start = int(line[x])
            end = int(line[x + 1])
            for timeslot in range(actionsize):
                if (timeslot * ratio <= start < (timeslot + 1) * ratio) or (start <= timeslot * ratio <= end) or (timeslot * ratio <= end < (timeslot + 1) * ratio):
                    a[timeslot][reg2idx[str(reg)]] = 125
                elif end > actionsize * ratio:
                    print "wrong"
                    sys.exit(0)
    physicalre(a, reward_dic, ratio, actionsize, slotstart, slotend, reg2idx)
    vrreward(a, vreward_dic, ratio, actionsize, slotstart, slotend, reg2idx)
    if not os.path.exists(directory):
        os.makedirs(directory) 
    if tofile:
        name = "./data/log/filename" + str(iteration) + ".png"
        plt.imshow(a, interpolation='nearest')
        plt.xticks(np.arange(0.0, 45, 1), np.arange(0, 45, 5))
        plt.yticks(np.arange(0.0, 45, 1), np.arange(0, 45, 5))
        plt.savefig(name)
    filename = "./data/log/filename" + str(iteration) + ".txt"
    copy2(state, filename)
    reward_dic.update(vreward_dic)
    s = a.flatten()
    return s, reward_dic, a


