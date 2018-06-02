import gridInitialize
import random
import copy
class QLearning_Solver(object):
    def __init__(self, maze, display=False):
        self.origField = maze
        self.Qvalue = {}
        self.Field = copy.deepcopy(self.origField)
        self.alpha = 0.2
        self.gamma  = 0.9
        self.epsilon = 0.2
        self.steps = 0
        self.score = 0
        self.display = display
        # print(self.Field.movable_vec is self.origField.movable_vec)
        # print(self.origField.movable_vec)
    def qlearn(self, greedy_flg=False):
        self.Field = copy.deepcopy(self.origField)
        state = self.Field.randomPickStart()
        print('current qlearn movable vec:', self.Field.movable_vec)
        print(self.Field.maze.D)
        while True:
            if greedy_flg:
                self.steps += 1
                action = self.choose_action_greedy(state)
                print("current state: {0} -> action: {1} ".format(state, action))
                if self.display:
                    self.Field.display(action)
                reward, tf = self.Field.get_val(action)
                self.score =  self.score + reward
                print("current step: {0} \t score: {1}\n".format(self.steps, self.score))
                if tf == True:
                    print("Goal!")
                    break
            else:
                action = self.choose_action(state)
            if self.update_Qvalue(state, action):
                break
            else:
                state = action

    def update_Qvalue(self, state, action):
        Q_s_a = self.get_Qvalue(state, action)
        mQ_s_a = max([self.get_Qvalue(action, n_action) for n_action in self.Field.get_actions(action)])
        r_s_a, finish_flg = self.Field.get_val(action)
        q_value = Q_s_a + self.alpha * ( r_s_a +  self.gamma * mQ_s_a - Q_s_a)
        self.set_Qvalue(state, action, q_value)
        return finish_flg


    def get_Qvalue(self, state, action):
        # state = (state[0],state[1])
        # action = (action[0],action[1])
        try:
            return self.Qvalue[state][action]
        except KeyError:
            return 0.0

    def set_Qvalue(self, state, action, q_value):
        print(state)
        print(self.Qvalue)
        # state = (state[0],state[1])
        # action = (action[0],action[1])
        self.Qvalue.setdefault(state,{})
        self.Qvalue[state][action] = q_value

    def choose_action(self, state):
        if self.epsilon < random.random():
            return random.choice(self.Field.get_actions(state))
        else:
            return self.choose_action_greedy(state)

    def choose_action_greedy(self, state):
        best_actions = []
        max_q_value = -1000
        for a in self.Field.get_actions(state):
            q_value = self.get_Qvalue(state, a)
            if q_value > max_q_value:
                best_actions = [a,]
                max_q_value = q_value
            elif q_value == max_q_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def dump_Qvalue(self):
        print("##### Dump Qvalue #####")
        for i, s in enumerate(self.Qvalue.keys()):
            for a in self.Qvalue[s].keys():
                print("\t\tQ(s, a): Q(%s, %s): %s" % (str(s), str(a), str(self.Qvalue[s][a])))
            if i != len(self.Qvalue.keys())-1:
                print('\t----- next state -----')


# In[53]:


size = 10
barriar_rate = 0.1

maze = gridInitialize.maze()
maze_field = gridInitialize.Field(maze)

maze_field.display()



learning_count = 1000
QL_solver = QLearning_Solver(maze_field, display=True)
for i in range(learning_count):
    print('i!!!!!!',i)
    # print('movable_vec::::',QL_solver.Field.movable_vec)
    QL_solver.qlearn()

QL_solver.dump_Qvalue()


# In[54]:

QL_solver.qlearn(greedy_flg=True)