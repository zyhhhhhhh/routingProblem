
'''
Create a 2d grid, with some percentage of blockage, some percentage of node pairs, and among these node pairs,
some are already connected. The purpose for it is to mimic the the fractional replay of a game.
'''
import numpy as np
import random
import astar3
import copy
import matplotlib.pyplot as plt
# Python Imaging Library imports
import matplotlib
from showit import image, tile
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K

class maze:
    height = 40
    width = 40
    blockage = 0.01
    nodePairPercent = 0.01
    connectPercent = 0
    mazeBoard = np.zeros((height, width))
    totalLen = 0
    connectCnt = 0
    D = {}  # stores start:end pair
    def __init__(self, cp = 0): #the init function initialize a configuration of maze, the change will be reflected
        #on the mazeBoard and D
        self.connectPercent = cp
        # blockagePointCnt = int(self.blockage*self.height*self.width)
        blockagePointCnt = self.height
        blockagePoints = zip(random.sample(range(self.width), blockagePointCnt), \
                             random.sample(range(self.height), blockagePointCnt))
        #print(list(blockagePoints))
        for (x,y) in blockagePoints:
            self.mazeBoard[x][y] = 1
        # print(self.mazeBoard)

        '''
        for simplicity, random blockage and node pairs can have overlaps. This means that blockage can be fewer than actual.
        '''
        # nodePairCnt = int(self.nodePairPercent*self.height*self.width)
        nodePairCnt = int(self.height/6)
        totalNodePairs = list(zip(random.sample(range(self.width), nodePairCnt*2), \
                             random.sample(range(self.height), nodePairCnt*2)))
        for (x,y) in totalNodePairs:
            self.mazeBoard[x][y] = 1
        startNodes, endNodes = totalNodePairs[0:nodePairCnt],totalNodePairs[nodePairCnt:]
        for ind in range(len(startNodes)):
            self.D[startNodes[ind]] = endNodes[ind]
        # print(startNodes)
        # print(endNodes)
        print("The size of D is:",len(self.D))
        # i = image(self.mazeBoard)
        # plt.show()
    def selectStartEnd(self, randomState = 1, startPoint = (0,0)):
        start = random.choice(list(self.D.keys()))
        end = self.D[start]
        return start,end

    def connectPathAstar(self,s = 0,e = 0,selectState = 1):
        print('D:::::',self.D)
        if selectState == 1:  #random select
            s,e = self.selectStartEnd()
            del self.D[s]
            self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 0, 0
            path = astar3.find_path_astar(self.mazeBoard,s,e)
            if path == "NO WAY!":
                self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 1, 1
                print(s,e)
                return -1
            else:
                self.Mazeboard = astar3.markPath(self.mazeBoard,path,s)
                self.totalLen+=len(path)
                self.connectCnt+=1
            print(path)
        else:
            del self.D[s]
            self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 0, 0
            path = astar3.find_path_astar(self.mazeBoard, s, e)
            if path == "NO WAY!":
                self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 1, 1
                print(s, e)
                return -1
            else:
                self.Mazeboard = astar3.markPath(self.mazeBoard, path, s)
                self.totalLen += len(path)
                self.connectCnt += 1
            print(path)
        return len(path)
    def randomConnect(self):
        self.connectPathAstar()
    def randomConnectAll(self):
        while self.D:
            self.randomConnect()
    def drawBoard(self):
        i = image(self.Mazeboard)
        plt.show()

class Field(object):

    def __init__(self, maze):
        self.maze = maze
        self.start_point = list(maze.D.keys())
        self.goal_point = list(maze.D.values())
        self.movable_vec = self.start_point  #change to all possible actions
    def display(self):
        field_data = copy.deepcopy(self.maze.mazeBoard)
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):#change to return all possible new connections.
        return self.movable_vec

    def get_val(self, state):  #returns a reward and if the game is over. Here gameover state should be changed
        #to all connected or tried, reward is -length of the route, finish goal is higher. state is start end pair.
        #for state transition, or Q table, state and action must both be starting points, at the beginning starting point
        #can be chosen randomly, this will give us a steady state transition
        s= state
        e = self.maze.D[s]
        rew = self.maze.connectPathAstar(s,e,selectState = 0)
        if rew == -1:  #no path, minus something larger than path
            v = -100
        else:
            v = -rew
        print(self.movable_vec)
        print('removing',state)
        self.movable_vec.remove(state)
        if self.movable_vec==[]:
            return v+2000, True
        else:
            return v, False
    def randomPickStart(self):
        return self.movable_vec.pop(np.random.randint(0,len(self.movable_vec)))
# M = maze()
# # s,e = M.selectStartEnd()
# # print(s,e)
# # print(M.mazeBoard)
# # board = copy.deepcopy(M.mazeBoard)
# # board[s[0]][s[1]],board[e[0],e[1]] = 0,0
# # print(e)
# # path = astar3.find_path_astar(board,s,e)
# # board = astar3.markPath(board, path, s)
# # print(path)
# # print(board)
# M.randomConnectAll()
# print(M.mazeBoard)
# print(M.totalLen/M.connectCnt)
# M.drawBoard()
