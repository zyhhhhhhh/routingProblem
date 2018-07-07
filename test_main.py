from __future__ import print_function
import gridInitialize
import dqn_solver
# import q_learning_solver
from tqdm import tqdm
import copy
import numpy as np
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
for i in range(3):
	state_size = 2
	action_size = 2
	dql_solver = dqn_solver.DQN_Solver(state_size, action_size)

	# episodes = 20000
	episodes = 20000
	times = 1000
	size = 10
	barriar_rate = 0.1

	# maze = gridInitialize.maze()
	maze_field = gridInitialize.Field()
	# maze_field.display()
	# print('movable vec: ',maze_field.movable_vec)
	# print('D: ',maze_field.D)
	for e in tqdm(range(episodes)):
		trainField = gridInitialize.copyMaze(maze_field)
		state = trainField.randomPickStart()
		score = 0
		for time in range(times):
			movables = copy.deepcopy(trainField.get_actions(state))
			action = copy.deepcopy(dql_solver.choose_action(state, movables))
			reward, done = copy.deepcopy(trainField.get_val(action))
			score = copy.deepcopy(score + reward)
			next_state = copy.deepcopy(action)
			next_movables = copy.deepcopy(trainField.get_actions(next_state))
			dql_solver.remember_memory(state, action, reward, next_state, next_movables, done)
			if done or time == (times - 1):
				# if e % 500 == 0:
				# 	print("episode: {}/{}, score: {}, e: {:.2} \t @ {}"
				# 		  .format(e, episodes, score, dql_solver.epsilon, time))
				break
			state = next_state
		dql_solver.replay_experience(32)

	# In[58]:
	testIterations = 5
	dqnReward0 = []
	dqnConnectivity0 = []
	dqnLen0 = []
	for _ in range(testIterations):
		testMaze = gridInitialize.copyMaze(maze_field)

		state = testMaze.randomPickStart()
		score = 0
		steps = 0
		while True:
			steps += 1
			movables = testMaze.get_actions(state)
			# print(state,movables)
			action = dql_solver.choose_best_action(state, movables)
			# print("current state: {0} -> action: {1} ".format(state, action))
			reward, done = testMaze.get_val(action)
			# maze_field.display()
			score = score + reward
			state = action
			# print("current step: {0} \t score: {1}\n".format(steps, score))
			if done:
				# testMaze.display()
				testMaze.connectCnt+=1
				# dqnReward0.append(testMaze.totalReward)
				dqnConnectivity0.append(testMaze.showConnectivity())
				dqnLen0.append(testMaze.totalLen/testMaze.connectCnt)
				break
	# dqnReward = np.mean(dqnReward0)
	dqnConnectivity = np.mean(dqnConnectivity0)
	dqnLen = np.mean(dqnLen0)
	print('all dqn connectivity: ', dqnConnectivity0)
	print("DQN mean connectivity: ", dqnConnectivity)
	# print('Reward is: ', dqnReward)
	print('length: ', dqnLen)

	# simpleReward0 = []
	simpleConnectivity0 = []
	simpleLen0 = []
	for _ in range(testIterations):
		simpleCopy = gridInitialize.copyMaze(maze_field)
		simpleCopy.randomConnectAll()
		# mazeCopy.totalReward += mazeCopy.rewardFinish
		# simpleReward0.append(simpleCopy.totalReward)
		simpleConnectivity0.append(simpleCopy.showConnectivity())
		simpleLen0.append(simpleCopy.totalLen/simpleCopy.connectCnt)
	# simpleReward = np.mean(simpleReward0)
	simpleConnectivity = np.mean(simpleConnectivity0)
	simpleLen = np.mean(simpleLen0)
	print('all heuristic0 connectivity: ',simpleConnectivity0)
	print('heuristic0 mean connectivity: ',simpleConnectivity)
	# print('Reward is: ', randomReward)
	print('length: ',simpleLen)

	# simple1Reward0 = []
	simple1Connectivity0 = []
	simple1Len0 = []
	for _ in range(testIterations):
		simple1Copy = gridInitialize.copyMaze(maze_field)
		simple1Copy.randomConnectAll()
		# mazeCopy.totalReward += mazeCopy.rewardFinish
		# simple1Reward0.append(simple1Copy.totalReward)
		simple1Connectivity0.append(simple1Copy.showConnectivity())
		simple1Len0.append(simple1Copy.totalLen/simple1Copy.connectCnt)
	# simple1Reward = np.mean(simple1Reward0)
	simple1Connectivity = np.mean(simple1Connectivity0)
	simple1Len = np.mean(simple1Len0)
	print('all heuristic1 connectivity: ',simple1Connectivity0)
	print('heuristic1 mean connectivity: ',simple1Connectivity)
	# print('Reward is: ', randomReward)
	print('length: ',simple1Len)

	randomReward0 = []
	randomConnectivity0 = []
	randomLen0 = []
	for _ in range(testIterations):
		mazeCopy = gridInitialize.copyMaze(maze_field)
		mazeCopy.randomConnectAll()
		# mazeCopy.totalReward += mazeCopy.rewardFinish
		randomReward0.append(mazeCopy.totalReward)
		randomConnectivity0.append(mazeCopy.showConnectivity())
		randomLen0.append(mazeCopy.totalLen/mazeCopy.connectCnt)
	randomReward = np.mean(randomReward0)
	randomConnectivity = np.mean(randomConnectivity0)
	randomLen = np.mean(randomLen0)
	print('all random connectivity: ',randomConnectivity0)
	print('random mean connectivity: ',randomConnectivity)
	# print('Reward is: ', randomReward)
	print('length: ',randomLen)

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

print('DQN Connectivity: ',allConnectivities0)
print('Random Connectivity: ',allConnectivities1)
print('Heuristic0 Connectivity: ', allConnectivities2)
print('Heuristic1 Connectivity: ', allConnectivities3)
print('DQN Length: ',allLengths0)
print('Random Length: ', allLengths1)
print('Heuristic0 Length: ',allLengths2)
print('Heuristic1 Length: ', allLengths3)
# print('DQN Score: ',allScores0)
# print('Random Score: ',allScores1)

print('mean connectivity: ',np.mean(np.array(allConnectivities0)),np.mean(np.array(allConnectivities1)),np.mean(np.array(allConnectivities2)),np.mean(np.array(allConnectivities3)))
print('mean length: ',np.mean(np.array(allLengths0)),np.mean(np.array(allLengths1)),np.mean(np.array(allLengths2)),np.mean(np.array(allLengths3)))