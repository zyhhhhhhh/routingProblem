from __future__ import print_function
import grayScaleGridInit
import newDQN_solver
# import q_learning_solver
from tqdm import tqdm
import copy
import numpy as np
import sys
from keras.models import model_from_json
import os
import csv
import gridInitialize
import numpy as np
import pandas as pds
import random
import copy
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam, RMSprop
from collections import deque
from keras import backend as K
import os
from random import sample as rsample
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten
from keras.optimizers import SGD, RMSprop
from matplotlib import pyplot as plt

allConnectivities0 = []
allLengths0 = []
allScores0 = []
allConnectivities1 = []
allLengths1 = []
allScores1 = []
allConnectivities2 = []
allLengths2 = []
allScores2 = []
allConnectivities3 = []
allLengths3 = []
allScores3 = []
def print(*args):
    __builtins__.print(*("%.2f" % a if isinstance(a, float) else a
                         for a in args))


state_size = 2
action_size = 2
if os.path.isfile('/Users/zyh/Documents/routingProblem/savedWeights/weights.h5'):
    dql_solver = newDQN_solver.DQN_Solver(state_size, action_size, new=False)
else:
    dql_solver = newDQN_solver.DQN_Solver(state_size, action_size,new = True)
    fields = ['dqn', 'random', 'h0','h1']
    with open(r'./results/test1.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

# dql_solver = newDQN_solver.DQN_Solver(state_size, action_size)

for i in range(10):

    # episodes = 20000
    trainAmount = 1000  #how many maze do we train on?
    episodes = 20  # how many episodes each maze?
    times = 1000
    size = 10
    barriar_rate = 0.1
    for _ in tqdm(range(trainAmount)):
        maze_field = grayScaleGridInit.Field()
        for e in range(episodes):
            trainField = grayScaleGridInit.copyMaze(maze_field)
            state = trainField.randomPickStart()
            score = 0
            for time in range(times):
                startpoints = copy.deepcopy(trainField.start_point)
                endpoints = copy.deepcopy(trainField.goal_point)
                connectedStart = copy.deepcopy(trainField.connectedStartPoints)
                connectedEnd = copy.deepcopy(trainField.connectedEndPoints)
                movables = copy.deepcopy(trainField.get_actions(state))
                action = copy.deepcopy(dql_solver.choose_action(startpoints, endpoints,\
                                                                connectedStart,connectedEnd, state, movables))
                reward, done = copy.deepcopy(trainField.get_val(action))
                score = copy.deepcopy(score + reward)
                next_state = copy.deepcopy(action)
                next_movables = copy.deepcopy(trainField.get_actions(next_state))
                dql_solver.remember_memory(startpoints,endpoints,connectedStart,connectedEnd, \
                                           state, action, reward, next_state, next_movables, done)
                if done or time == (times - 1):
                    # if e % 500 == 0:
                    # 	print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
                    # 		  .format(e, episodes, score, dql_solver.epsilon, time))
                    break
                state = next_state
            # print(trainField.connectedStartPoints)
            # print(trainField.connectedEndPoints)
            dql_solver.replay_experience(32)


    testIterations = 5
    dqnConnectivity0 = []
    dqnLen0 = []
    randomConnectivity0 = []
    randomLen0 = []
    simpleConnectivity0 = []
    simpleLen0 = []
    simple1Connectivity0 = []
    simple1Len0 = []
    for _ in range(testIterations):
        tmaze = grayScaleGridInit.Field()


        #dqn
        testMaze = grayScaleGridInit.copyMaze(tmaze)
        state = testMaze.randomPickStart()
        score = 0
        steps = 0
        while True:
            steps += 1
            movables = testMaze.get_actions(state)
            # print(state,movables)
            action = dql_solver.choose_best_action(testMaze.start_point, testMaze.goal_point,\
                                                   testMaze.connectedStartPoints, testMaze.connectedEndPoints,\
                                                   state, movables)
            # print("current state: {0} -> action: {1} ".format(state, action))
            reward, done = testMaze.get_val(action)
            # maze_field.display()
            score = score + reward
            state = action
            # print("current step: {0} \t score: {1}\n".format(steps, score))
            if done:
                # testMaze.display()
                testMaze.connectCnt += 1
                # dqnReward0.append(testMaze.totalReward)
                dqnConnectivity0.append(testMaze.showConnectivity())
                dqnLen0.append(testMaze.totalLen / testMaze.connectCnt)
                break
                # dqnReward = np.mean(dqnReward0)

        #random
        mazeCopy = grayScaleGridInit.copyMaze(tmaze)
        mazeCopy.randomConnectAll()
        randomConnectivity0.append(mazeCopy.showConnectivity())
        randomLen0.append(mazeCopy.totalLen / mazeCopy.connectCnt)

        #H0
        simpleCopy = grayScaleGridInit.copyMaze(tmaze)
        simpleCopy.heuristicConnectAll(state = 0)
        simpleConnectivity0.append(simpleCopy.showConnectivity())
        simpleLen0.append(simpleCopy.totalLen / simpleCopy.connectCnt)

        #H1
        simple1Copy = grayScaleGridInit.copyMaze(tmaze)
        simple1Copy.heuristicConnectAll(state = 1)
        simple1Connectivity0.append(simple1Copy.showConnectivity())
        simple1Len0.append(simple1Copy.totalLen / simple1Copy.connectCnt)



    #DQN
    dqnConnectivity = np.mean(dqnConnectivity0)
    dqnLen = np.mean(dqnLen0)
    print('all dqn connectivity: ', dqnConnectivity0)
    print("DQN mean connectivity: ", dqnConnectivity)
    print('length: ', dqnLen)

    #random
    randomConnectivity = np.mean(randomConnectivity0)
    randomLen = np.mean(randomLen0)
    print('all random connectivity: ', randomConnectivity0)
    print('random mean connectivity: ', randomConnectivity)
    print('length: ', randomLen)

    #h0
    simpleConnectivity = np.mean(simpleConnectivity0)
    simpleLen = np.mean(simpleLen0)
    print('all heuristic0 connectivity: ',simpleConnectivity0)
    print('heuristic0 mean connectivity: ',simpleConnectivity)
    print('length: ',simpleLen)

    # H1
    simple1Connectivity = np.mean(simple1Connectivity0)
    simple1Len = np.mean(simple1Len0)
    print('all heuristic1 connectivity: ',simple1Connectivity0)
    print('heuristic1 mean connectivity: ',simple1Connectivity)
    print('length: ',simple1Len)



    allConnectivities0.append(dqnConnectivity)
    allLengths0.append(dqnLen)
    # allScores0.append(dqnReward)

    allConnectivities1.append(randomConnectivity)
    allLengths1.append(randomLen)
    # allScores1.append(randomReward)

    allConnectivities2.append(simpleConnectivity)
    allLengths2.append(simpleLen)
    # allScores1.append(simpleReward)

    allConnectivities3.append(simple1Connectivity)
    allLengths3.append(simple1Len)
    # allScores1.append(simple1Reward)
    # add to csv:
    fields = [dqnConnectivity, randomConnectivity,
              simpleConnectivity, simple1Connectivity]
    with open(r'./results/test1.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

print('mean connectivity: ',np.mean(np.array(allConnectivities0)),np.mean(np.array(allConnectivities1)),np.mean(np.array(allConnectivities2)),np.mean(np.array(allConnectivities3)))
print('mean length: ',np.mean(np.array(allLengths0)),np.mean(np.array(allLengths1)),np.mean(np.array(allLengths2)),np.mean(np.array(allLengths3)))

# serialize model to JSON
model_json = dql_solver.model.to_json()
with open("./savedWeights/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
dql_solver.model.save_weights("./savedWeights/weights.h5")
print("Saved model to disk")