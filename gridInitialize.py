
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

class maze:
    height = 20
    width = 20
    blockage = 0.05
    nodePairPercent = 0.02
    connectPercent = 0
    mazeBoard = np.zeros((height, width))
    D = {}  # stores start:end pair
    def __init__(self, cp = 0): #the init function initialize a configuration of maze, the change will be reflected
        #on the mazeBoard and D
        self.connectPercent = cp
        blockagePointCnt = int(self.blockage*self.height*self.width)
        blockagePoints = zip(random.sample(range(self.width), blockagePointCnt), \
                             random.sample(range(self.height), blockagePointCnt))
        #print(list(blockagePoints))
        for (x,y) in blockagePoints:
            self.mazeBoard[x][y] = 1
        # print(self.mazeBoard)

        '''
        for simplicity, random blockage and node pairs can have overlaps. This means that blockage can be fewer than actual.
        '''
        nodePairCnt = int(self.nodePairPercent*self.height*self.width)
        totalNodePairs = list(zip(random.sample(range(self.width), nodePairCnt*2), \
                             random.sample(range(self.height), nodePairCnt*2)))
        for (x,y) in totalNodePairs:
            self.mazeBoard[x][y] = 1
        startNodes, endNodes = totalNodePairs[0:nodePairCnt],totalNodePairs[nodePairCnt:]
        for ind in range(len(startNodes)):
            self.D[startNodes[ind]] = endNodes[ind]
        # print(startNodes)
        # print(endNodes)
        # print(self.D)
        # i = image(self.mazeBoard)
        # plt.show()
    def selectStartEnd(self, randomState = 1, startPoint = (0,0)):
        start = random.choice(list(self.D.keys()))
        end = self.D[start]
        return start,end

M = maze()
s,e = M.selectStartEnd()
print(s,e)
print(M.mazeBoard)
board = copy.deepcopy(M.mazeBoard)
board[s[0]][s[1]],board[e[0],e[1]] = 0,0
path = astar3.find_path_astar(board,s,e)
board = astar3.markPath(board, path, s)
print(path)
print(board)
