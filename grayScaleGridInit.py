
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
import matplotlib.image as im
from showit import image, tile
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K
#
# class maze:
#     height = 100
#     width = 100
#     blockage = 0.01
#     nodePairPercent = 0.01
#     connectPercent = 0
#     mazeBoard = np.zeros((height, width))
#     totalLen = 0
#     connectCnt = 0
#     nodePairCnt = int(height / 6)
#     D = {}  # stores start:end pair
#     def __init__(self, cp = 0): #the init function initialize a configuration of maze, the change will be reflected
#         #on the mazeBoard and D
#         self.connectPercent = cp
#         # blockagePointCnt = int(self.blockage*self.height*self.width)
#         blockagePointCnt = self.height
#         blockagePoints = zip(random.sample(range(self.width), blockagePointCnt), \
#                              random.sample(range(self.height), blockagePointCnt))
#         #print(list(blockagePoints))
#         for (x,y) in blockagePoints:
#             self.mazeBoard[x][y] = 1
#         # print(self.mazeBoard)
#
#         '''
#         for simplicity, random blockage and node pairs can have overlaps. This means that blockage can be fewer than actual.
#         '''
#         # nodePairCnt = int(self.nodePairPercent*self.height*self.width)
#         totalNodePairs = list(zip(random.sample(range(self.width), self.nodePairCnt*2), \
#                              random.sample(range(self.height), self.nodePairCnt*2)))
#         for (x,y) in totalNodePairs:
#             self.mazeBoard[x][y] = 1
#         startNodes, endNodes = totalNodePairs[0:self.nodePairCnt],totalNodePairs[self.nodePairCnt:]
#         for ind in range(len(startNodes)):
#             self.D[startNodes[ind]] = endNodes[ind]
#         # print(startNodes)
#         # print(endNodes)
#         print("The size of D is:",len(self.D))
#         # i = image(self.mazeBoard)
#         # plt.show()
#
#
#     def selectStartEnd(self, randomState = 1, startPoint = (0,0)):
#         start = random.choice(list(self.D.keys()))
#         end = self.D[start]
#         return start,end
#
#     def connectPathAstar(self,s = 0,e = 0,selectState = 1):
#         print('D:::::',self.D)
#         if selectState == 1:  #random select
#             s,e = self.selectStartEnd()
#             del self.D[s]
#             self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 0, 0
#             path = astar3.find_path_astar(self.mazeBoard,s,e)
#             if path == "NO WAY!":
#                 self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 1, 1
#                 print(s,e)
#                 return -1
#             else:
#                 self.Mazeboard = astar3.markPath(self.mazeBoard,path,s)
#                 self.totalLen+=len(path)
#                 self.connectCnt+=1
#             print(path)
#         else:
#             del self.D[s]
#             self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 0, 0
#             path = astar3.find_path_astar(self.mazeBoard, s, e)
#             if path == "NO WAY!":
#                 self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 1, 1
#                 print(s, e)
#                 return -1
#             else:
#                 self.Mazeboard = astar3.markPath(self.mazeBoard, path, s)
#                 self.totalLen += len(path)
#                 self.connectCnt += 1
#             print(path)
#         return len(path)
#     def randomConnect(self):
#         self.connectPathAstar()
#     def randomConnectAll(self):
#         while self.D:
#             self.randomConnect()
#     def drawBoard(self):
#         i = image(self.Mazeboard)
#         plt.show()

class Field(object):
    # self.height = 100
    # self.width = 100
    # self.blockage = 0.01
    # self.penaltyNoConnection = -4*height
    # self.nodePairCnt = int(height/25)
    # self.rewardFinish = 2*height*nodePairCnt
    # self.nodePairPercent = 0.01
    # self.connectPercent = 0
    # self.mazeBoard = np.zeros((height, width))
    # self.totalLen = 0
    # self.connectCnt = 0
    # self.totalReward = 0
    # self.D = {}  # stores start:end pair
    def __init__(self, cp = 0):#the init function initialize a configuration of maze, the change will be reflected
        #on the mazeBoard and D
        self.connectPercent = cp
        self.height = 100
        self.width = 100
        self.blockage = 0.01
        self.penaltyNoConnection = -6 * self.height   #whatabout just -1 and 1
        self.nodePairCnt = 40
        self.rewardFinish = 4 * self.height * self.nodePairCnt
        self.nodePairPercent = 0.01
        self.connectPercent = 0
        self.mazeBoard = np.zeros((self.height, self.width))
        self.totalLen = 0
        self.connectCnt = 0
        self.totalReward = 0

        self.D = {}  # stores start:end pair
        # blockagePointCnt = int(self.blockage*self.height*self.width)
        # blockagePointCnt = self.height
        # blockagePoints = zip(random.sample(range(self.width), blockagePointCnt), \
        #                      random.sample(range(self.height), blockagePointCnt))
        # #print(list(blockagePoints))
        # for (x,y) in blockagePoints:
        #     self.mazeBoard[x][y] = 1
        # print(self.mazeBoard)

        '''
        for simplicity, random blockage and node pairs can have overlaps. This means that blockage can be fewer than actual.
        '''
        randomSamples = np.random.choice(self.height*self.width,self.nodePairCnt*2)
        totalNodePairs = [(int(x/self.width),int(x%self.width)) for x in randomSamples]
        # totalNodePairs = list(zip(random.sample(range(self.width), self.nodePairCnt*2), \
        #                      random.sample(range(self.height), self.nodePairCnt*2)))

        startNodes, endNodes = totalNodePairs[0:self.nodePairCnt],totalNodePairs[self.nodePairCnt:]
        for (x,y) in startNodes:
            self.mazeBoard[x][y] = 255
        for (x,y) in endNodes:
            self.mazeBoard[x][y] = 155
        for ind in range(len(startNodes)):
            self.D[startNodes[ind]] = endNodes[ind]
        # print(startNodes)
        # print(endNodes)
        # print("D is:",self.D)
        # i = image(self.mazeBoard)
        # plt.show()
        self.start_point = list(self.D.keys())
        self.goal_point = list(self.D.values())
        self.movable_vec = self.start_point  #change to all possible actions

    def showConnectivity(self):
        return (self.connectCnt)/self.nodePairCnt

    def display(self):
        field_data = copy.deepcopy(self.mazeBoard)
        for line in field_data:
                print ("\t" + "%3s " * len(line) % tuple(line))

    def get_actions(self, state):#change to return all possible new connections.
        return self.movable_vec

    def get_val(self, state):  #returns a reward and if the game is over. Here gameover state should be changed
        #to all connected or tried, reward is -length of the route, finish goal is higher. state is start end pair.
        #for state transition, or Q table, state and action must both be starting points, at the beginning starting point
        #can be chosen randomly, this will give us a steady state transition
        s= state
        e = self.D[s]
        rew = self.connectPathAstar(s,e,selectState = 0)
        if rew == -1:  #no path, minus something larger than path
            # v = self.penaltyNoConnection
            v = -1
        else:
            # v = -rew
            v = 1
        # print('size of vec:',len(self.movable_vec))
        # print('removing',state)
        self.movable_vec.remove(state)
        if self.movable_vec==[]:
            return v, True
        else:
            return v, False
    def randomPickStart(self):
        return self.movable_vec.pop(np.random.randint(0,len(self.movable_vec)))
    def selectStartEnd(self, randomState = 1, startPoint = (0,0)):
        start = random.choice(list(self.D.keys()))
        end = self.D[start]
        return start,end

    def connectPathAstar(self,s = 0,e = 0,selectState = 1):
        # print('D:::::',len(self.D))
        if selectState == 1:  #random select
            s,e = self.selectStartEnd()
            del self.D[s]
            self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 0, 0
            path = astar3.find_path_astar(self.mazeBoard,s,e)
            if path == "NO WAY!":
                self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 1, 1
                # print('NO Way!!!!!!!', s, e)
                # self.totalReward += self.penaltyNoConnection
                self.totalReward -=1
                return -1
            else:
                self.Mazeboard = astar3.markPath(self.mazeBoard,path,s)
                self.totalLen+=len(path)
                self.connectCnt+=1
                # self.totalReward -= len(path)
                self.totalReward += 1
                # print(len(path))
        else:
            del self.D[s]
            self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 0, 0
            path = astar3.find_path_astar(self.mazeBoard, s, e)
            if path == "NO WAY!":
                self.mazeBoard[s[0]][s[1]], self.mazeBoard[e[0], e[1]] = 1, 1
                # print('NO Way!!!!!!!', s, e)
                # self.totalReward += self.penaltyNoConnection
                self.totalReward -= 1
                return -1
            else:
                self.Mazeboard = astar3.markPath(self.mazeBoard, path, s)
                self.totalLen += len(path)
                self.connectCnt += 1
                # self.totalReward-=len(path)
                self.totalReward += 1
                # print(len(path))
        return len(path)
    def randomConnect(self):
        self.connectPathAstar()
    def randomConnectAll(self):
        while self.D:
            self.randomConnect()

    def heuristicConnectAll(self,state = 0):  #state = 0 is increasing order from small to big manhattan distance
        newStartPoints = sorted(self.start_point, key = lambda x:np.abs(x[0]-self.D[x][0])+np.abs(x[1]-self.D[x][1]))
        # print(self.D)
        # print(newStartPoints)

        if state == 1:
            newStartPoints = newStartPoints[::-1]
        t = newStartPoints[0]
        t1 = newStartPoints[1]
        # print(np.abs(t[0] - self.D[t][0])+np.abs(t[1]-self.D[t][1]))
        # print(np.abs(t1[0] - self.D[t1][0]) + np.abs(t1[1] - self.D[t1][1]))
        for s in newStartPoints:
            self.connectPathAstar(s,self.D[s],selectState=0)
    def drawBoard(self):
        i = image(self.Mazeboard)
        plt.show()

def copyMaze(maze_field):
    temp = Field()
    temp.D = dict()
    temp.D.update(maze_field.D)
    # temp.D = copy.deepcopy(maze_field.D)
    temp.mazeBoard = copy.deepcopy(maze_field.mazeBoard)
    temp.start_point = copy.deepcopy(maze_field.start_point)
    temp.goal_point = copy.deepcopy(maze_field.goal_point)
    temp.movable_vec = copy.deepcopy(maze_field.movable_vec)
    return temp



M = Field()
im.imsave('grayScaleGridInit.png', M.mazeBoard)
#
# M = Field()
# M.heuristicConnectAll(state = 1)
# # M.randomConnectAll()
# print(M.showConnectivity())
# M.display()
#
# M = Field()
# C = copyMaze(M)
# print(M.D)
# print(C.D)
# print(M.movable_vec)
# print(C.movable_vec)
# M.connectCnt = 3
# C.connectCnt = 4
# print(M.connectCnt,C.connectCnt)
# print(M.showConnectivity())
# print(C.showConnectivity())
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
