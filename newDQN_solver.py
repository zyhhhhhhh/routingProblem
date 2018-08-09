# coding: utf-8

# # Solving Maze with A-star algorithm, Q-learning and Deep Q-network

# ### Objective of this notebook is to solve self-made maze with A-star algorithm, Q-learning and Deep Q-network.
# ### The maze is in square shape, consists of start point, goal point and tiles in the mid of them.
# ### Each tile has numericals as its point. In other words, if you step on to the tile with -1, you get 1 point subtracted.
# ### The maze has blocks to prevent you from taking the route.

# In[1]:
import gridInitialize
import numpy as np
import pandas as pds
import random
import copy
from keras.models import model_from_json
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

class DQN_Solver:
    def __init__(self, state_size, action_size,new = True):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9
        self.epsilon = 0.2
        self.e_decay = 0.9999
        self.e_min = 0.01
        self.learning_rate = 0.0001
        self.model = self.build_model(new)

    def build_model(self, new):
        if new:
            model = Sequential()
            model.add(Dense(256, input_shape=(122,2), activation='tanh'))
            model.add(Flatten())
            model.add(Dense(512, activation='tanh'))
            model.add(Dense(512, activation='tanh'))
            model.add(Dense(256, activation='tanh'))
            model.add(Dense(256, activation='tanh'))
            model.add(Dense(1, activation='linear'))
            model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
            print("Building new model")
        else:
            json_file = open('./savedWeights/model.json', 'r')
            loaded_model_json = json_file.read()
            json_file.close()
            model = model_from_json(loaded_model_json)
            # load weights into new model
            model.load_weights("./savedWeights/weights.h5")
            model.compile(loss="mse", optimizer=RMSprop(lr=self.learning_rate))
            print("Loaded model from disk")
        return model

    def remember_memory(self, startpoints,endpoints,connectedStart,connectedEnd,state, action, reward, next_state, next_movables, done):
        self.memory.append((startpoints, endpoints,connectedStart,connectedEnd,state, action, reward, next_state, next_movables, done))

    def choose_action(self, startpoints,endpoints,connectedStart,connectedEnd, state, movables):
        if self.epsilon >= random.random():
            return random.choice(movables)
        else:
            return self.choose_best_action(startpoints,endpoints,connectedStart,connectedEnd, state, movables)

    def choose_best_action(self, startpoints,endpoints,connectedStart,connectedEnd, state, movables):
        best_actions = []
        max_act_value = -300
        for a in movables:
            np_action = self.formatinput(startpoints,endpoints,connectedStart,connectedEnd,state,a)
            if np_action.shape != (1,122,2):
                print (np_action.shape)
                print(state,a,len(startpoints),len(endpoints))
                break
            act_value = self.model.predict(np_action)
            if act_value > max_act_value:
                best_actions = [a, ]
                max_act_value = act_value
            elif act_value == max_act_value:
                best_actions.append(a)
        return random.choice(best_actions)
    def formatinput(self,startpoints,endpoints,connectedStart,connectedEnd,state,action):
        temp = startpoints[:]
        temp.extend(endpoints[:])
        temp.extend(connectedStart[:])
        temp.extend(connectedEnd[:])
        temp.append(state)
        temp.append(action)
        np_action = np.array([temp])
        return np_action
    def formatinputaction(self,startpoints,endpoints,connectedStart,connectedEnd,state,action):
        temp = startpoints[:]
        temp.extend(endpoints[:])
        temp.extend(connectedStart[:])
        temp.extend(connectedEnd[:])
        temp.append(state)
        temp.append(action)
        input_action = np.array(temp)
        return input_action
    def replay_experience(self, batch_size):
        batch_size = min(batch_size, len(self.memory))
        minibatch = random.sample(self.memory, batch_size)
        X = []
        Y = []
        for i in range(batch_size):
            startpoints,endpoints,connectedStart,connectedEnd,state, action, reward, next_state, next_movables, done = minibatch[i]
            # print(state,action,reward,next_state,next_movables,done)
            input_action = self.formatinputaction(startpoints,endpoints, connectedStart,connectedEnd,state, action)
            if done:
                target_f = reward
            else:
                next_rewards = []
                for i in next_movables:
                    np_next_s_a = self.formatinput(startpoints,endpoints,connectedStart,connectedEnd,next_state, i)
                    next_rewards.append(self.model.predict(np_next_s_a))
                np_n_r_max = np.amax(np.array(next_rewards))
                target_f = reward + self.gamma * np_n_r_max
            X.append(input_action)
            Y.append(target_f)
        np_X = np.array(X)
        np_Y = np.array([Y]).T
        self.model.fit(np_X, np_Y, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay


# testing

# state_size = 2
# action_size = 2
# dql_solver = DQN_Solver(state_size, action_size)
#
# episodes = 20000
# episodes = 1
# times = 1000
# size = 10
# barriar_rate = 0.1
#
# # maze = gridInitialize.maze()
# maze_field = gridInitialize.Field()
# # maze_field.display()
#
# for e in range(episodes):
#     Field = copy.deepcopy(maze_field)
#     Field.D = copy.deepcopy(maze_field.D)
#     Field.mazeBoard = copy.deepcopy(maze_field.mazeBoard)
#     Field.start_point = copy.deepcopy(maze_field.start_point)
#     Field.goal_point = copy.deepcopy(maze_field.goal_point)
#     Field.movable_vec = copy.deepcopy(maze_field.movable_vec)
#     state = Field.randomPickStart()
#     score = 0
#     for time in range(times):
#         movables = copy.deepcopy(Field.get_actions(state))
#         action = copy.deepcopy(dql_solver.choose_action(state, movables))
#         reward, done = copy.deepcopy(Field.get_val(action))
#         score = copy.deepcopy(score + reward)
#         next_state = copy.deepcopy(action)
#         next_movables = copy.deepcopy(Field.get_actions(next_state))
#         dql_solver.remember_memory(state, action, reward, next_state, next_movables, done)
#         if done or time == (times - 1):
#             if e % 500 == 0:
#                 print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
#                       .format(e, episodes, score, dql_solver.epsilon, time))
#             break
#         state = next_state
#     dql_solver.replay_experience(32)
#
# # In[58]:
#
#
# testMaze = gridInitialize.copyMaze(maze_field)
#
# state = testMaze.randomPickStart()
# score = 0
# steps = 0
# while True:
#     steps += 1
#     movables = testMaze.get_actions(state)
#     print(state,movables)
#     action = dql_solver.choose_best_action(state, movables)
#     print("current state: {0} -> action: {1} ".format(state, action))
#     reward, done = testMaze.get_val(action)
#     # maze_field.display()
#     score = score + reward
#     state = action
#     print("current step: {0} \t score: {1}\n".format(steps, score))
#     if done:
#         # testMaze.display()
#         print("DQN result: ", testMaze.totalReward+testMaze.rewardFinish)
#         print('connectivity is: ',testMaze.showConnectivity())
#         print('length: ',testMaze.totalLen)
#         break
#
#
# mazeCopy = gridInitialize.copyMaze(maze_field)
# mazeCopy.randomConnectAll()
# mazeCopy.totalReward += mazeCopy.rewardFinish
# print('random result: ',mazeCopy.totalReward)
# print('connectivity is: ',mazeCopy.showConnectivity())
# print('length: ',mazeCopy.totalLen)
# In[ ]:




# In[ ]:



