import parse
from subprocess import call
import cv2
import numpy as np
from skimage.transform import resize
from shutil import copy2
import os


def teststate():
    g = {'39': 0, '38': 1, '37': 2, '43': 3, '40': 4, '115': 5, '114': 6, '117': 7, '116': 8, '111': 9, '35': 13, '113': 10, '112': 11, '118': 12, '229': 14, '237': 15, '245': 16}
    res, _ = parse.getstate("teststate.txt", 1, 300, 17, g)
    print res

def genphys(regs):
    string = ""
    for i in range(len(regs)):
        string = string  + str(regs[i]) + "&" + str(i) + "&"
    string = string + "\n"
    return string
def genoccupy(regs, start, end):
    string = str(regs) + "&" + str(start) + "&" + str(end)
    string = string + "\n"
    return string

def normal():
    f = open('workfile.txt', 'w')
    start = 0
    end = 2
    firstline = "3333&" + str(start) + "&" + str(end) + "\n"
    f.writelines(firstline)
    f.writelines("reward\n")
    phys = [3, 4]
    physline = genphys(phys)
    f.write(physline)
    vrs = [0]
    f.writelines("vreward\n")
    virtline = genphys(vrs)
    f.writelines(virtline)
    occ = [1, 2]
    for x in range(len(occ)):
        f.write(genoccupy(occ[x], 3, 4))
    f.close()
    reg2idx = {"0":0, "1": 1, "2": 2, "3": 3, "4": 4}
    a, y, h = parse.getstate("workfile.txt", 0, 5, 5, reg2idx)
    return h

def test_normal(a):
    for i in range(5):
        for j in range(5):
            if 0 <= i <= 2:
                if j == 0:
                    if a[i][j] != 75:
                        print "falut"
                if j == 3 or j == 4:
                    if a[i][j] != 255:
                        print "falut"
            if 3 <= i <= 4:
                if j == 1 or j == 2:
                    if a[i][j] != 125:
                        print "falut"
                        
def physicaldownonoccupy():
    f = open('workfile.txt', 'w')
    start = 2
    end = 3
    firstline = "3333&" + str(start) + "&" + str(end) + "\n"
    f.writelines(firstline)
    f.writelines("reward\n")
    phys = [2, 3]
    physline = genphys(phys)
    f.write(physline)
    vrs = [0]
    f.writelines("vreward\n")
    virtline = genphys(vrs)
    f.writelines(virtline)
    occ = [2, 3]
    for x in range(len(occ)):
        f.write(genoccupy(occ[x], 1, 2))
    f.close()
    reg2idx = {"0":0, "1": 1, "2": 2, "3": 3, "4": 4}
    a, y, h = parse.getstate("workfile.txt", 0, 5, 5, reg2idx)
    return h

def test_physical_down(a):
    for time in range(5):
        for reg in range(5):
            if 2 <= time <= 3:
                if reg == 0:
                    if a[time][reg] != 75:
                        print "vir falut"
                if reg == 2 or reg == 3:
                    if a[time][reg] != 255:
                        print "phys falut"
            if 1 <= time <= 1:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 125:
                        print "occu falut"

def virtualdownonoccpy():
    f = open('workfile.txt', 'w')
    start = 2
    end = 3
    firstline = "3333&" + str(start) + "&" + str(end) + "\n"
    f.writelines(firstline)
    f.writelines("reward\n")
    phys = [1]
    physline = genphys(phys)
    f.write(physline)
    vrs = [2, 3]
    f.writelines("vreward\n")
    virtline = genphys(vrs)
    f.writelines(virtline)
    occ = [2, 3]
    for x in range(len(occ)):
        f.write(genoccupy(occ[x], 1, 2))
    f.close()
    reg2idx = {"0":0, "1": 1, "2": 2, "3": 3, "4": 4}
    a, y, h = parse.getstate("workfile.txt", 0, 5, 5, reg2idx)
    return h

def test_virtual_down(a):
    for time in range(5):
        for reg in range(5):
            if 2 <= time <= 3:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 75:
                        print "vir falut"
                if reg == 1:
                    if a[time][reg] != 255:
                        print "phys falut"
            if 1 <= time <= 1:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 125:
                        print "occu falut"
                        
                        
                        
def virtualuponoccupy():
    f = open('workfile.txt', 'w')
    start = 0
    end = 2
    firstline = "3333&" + str(start) + "&" + str(end) + "\n"
    f.writelines(firstline)
    f.writelines("reward\n")
    phys = [1]
    physline = genphys(phys)
    f.write(physline)
    vrs = [2, 3]
    f.writelines("vreward\n")
    virtline = genphys(vrs)
    f.writelines(virtline)
    occ = [2, 3]
    for x in range(len(occ)):
        f.write(genoccupy(occ[x], 2, 3))
    f.close()
    reg2idx = {"0":0, "1": 1, "2": 2, "3": 3, "4": 4}
    a, y, h = parse.getstate("workfile.txt", 0, 5, 5, reg2idx)
    return h
def test_virtual_up(a):
    for time in range(5):
        for reg in range(5):
            if 0 <= time <= 2:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 75:
                        print "vir falut"
                if reg == 1:
                    if a[time][reg] != 255:
                        print "phys falut"
            if 3 <= time <= 3:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 125:
                        print "occu falut"

def physicaluponoccupy():
    f = open('workfile.txt', 'w')
    start = 0
    end = 2
    firstline = "3333&" + str(start) + "&" + str(end) + "\n"
    f.writelines(firstline)
    f.writelines("reward\n")
    phys = [1]
    physline = genphys(phys)
    f.write(physline)
    vrs = [2, 3]
    f.writelines("vreward\n")
    virtline = genphys(vrs)
    f.writelines(virtline)
    occ = [1]
    for x in range(len(occ)):
        f.write(genoccupy(occ[x], 2, 3))
    f.close()
    reg2idx = {"0":0, "1": 1, "2": 2, "3": 3, "4": 4}
    a, y, h = parse.getstate("workfile.txt", 0, 5, 5, reg2idx)
    return h


def test_physical_up(a):
    for time in range(5):
        for reg in range(5):
            if 0 <= time <= 2:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 75:
                        print "vir falut"
                if reg == 1:
                    if a[time][reg] != 255:
                        print "phys falut"
            if 3 <= time <= 3:
                if reg == 1:
                    if a[time][reg] != 125:
                        print "occu falut"
                        
def virtualoverlapoccupy():
    f = open('workfile.txt', 'w')
    start = 1
    end = 2
    firstline = "3333&" + str(start) + "&" + str(end) + "\n"
    f.writelines(firstline)
    f.writelines("reward\n")
    phys = [1]
    physline = genphys(phys)
    f.write(physline)
    vrs = [2, 3]
    f.writelines("vreward\n")
    virtline = genphys(vrs)
    f.writelines(virtline)
    occ = [2, 4]
    for x in range(len(occ)):
        f.write(genoccupy(occ[x], 0, 4))
    f.close()
    reg2idx = {"0":0, "1": 1, "2": 2, "3": 3, "4": 4}
    a, y, h = parse.getstate("workfile.txt", 0, 5, 5, reg2idx)
    return h
    
    
def test_virtual_overlap(a):
    for time in range(5):
        for reg in range(5):
            if 1 <= time <= 2:
                if reg == 2 or reg == 3:
                    if a[time][reg] != 75:
                        print "vir falut"
                if reg == 1:
                    if a[time][reg] != 255:
                        print "phys falut"
            if 0 <= time <= 4:
                if reg == 4:
                    if a[time][reg] != 125:
                        print "f occu falut"
                elif reg == 2:
                    if time != 1 and time != 2:
                        if a[time][reg] != 125:
                            print "occu falut"

def main():
    a = normal()
    test_normal(a)
    b = physicaldownonoccupy()
    test_physical_down(b)
    c = virtualdownonoccpy()
    test_virtual_down(c)
    d = virtualuponoccupy()
    test_virtual_up(d)
    e = physicaluponoccupy()
    test_physical_up(e)
    f = virtualoverlapoccupy()
    test_virtual_overlap(f)



if __name__ == "__main__":
        main()
