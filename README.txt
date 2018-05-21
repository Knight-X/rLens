rlens: compiler optimization framework


Introduction

rLens is an compiler optimization framework via reinforcement learning. The project could optimize register allocation for graph computing like problem in different environment, such as compiler, deep learning framework. 


Register Allocation:
  1. allocate virtual register to physical register
  2. traditionally, it was np-complete problem with graph-color strategy



usage:
  python train_pg.py -h for more information


Result:
  ![Alt text](./pics/my_loss.png "Optional title")
reference:
  https://github.com/berkeleydeeprlcourse/homework 
