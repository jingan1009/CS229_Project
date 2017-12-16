import numpy as np
import random
from collections import deque

class ReplayMemoryQueue(object):
    def __init__(self,buffer_size):
        # Buffer includes state, action, reward, next_states, legal actions of next state
        self.buffer = deque()
        self.buffer_size = buffer_size    
        self.states_t = deque()
                
    def sample(self, batch_size):
        # assert batch_size <= len(self.buffer)
        indices        = np.random.choice(len(self.buffer),batch_size)
        state_t_batch = []
        action_batch = []
        reward_batch = []
        state_t1_batch = []
        terminal_batch = []
        legal_actions_batch = []


        for i in indices:
            state_t_batch.append(self.buffer[i][0])
            action_batch.append(self.buffer[i][1])
            reward_batch.append(self.buffer[i][2] )       
            state_t1_batch.append(self.buffer[i][3])
            terminal_batch.append(self.buffer[i][4])
            legal_actions_batch.append(self.buffer[i][5])

        state_t_batch = np.array(state_t_batch)
        action_batch = np.array(action_batch)
        reward_batch = np.array(reward_batch)
        state_t1_batch = np.array(state_t1_batch)
        terminal_batch = np.array(terminal_batch)
        legal_actions_batch = np.array(legal_actions_batch)



        return state_t_batch, action_batch, terminal_batch, state_t1_batch, reward_batch, legal_actions_batch

    # store all data using one function
    def storeAll(self,state, action, reward, state_next, terminal, legalactions):
        #  Action 
        if action == 'North':
            action=0
        elif action == 'East':
            action=1
        elif action == 'South':
            action=2
        elif action == 'West':
            action=3
        elif action == 'Stop':
            action=4

        # Legal actions
        legal_actions= np.zeros((5))
        legal_actions[0] = 1 if 'North' in legalactions else 0
        legal_actions[1] = 1 if 'East' in legalactions else 0
        legal_actions[2] = 1 if 'South' in legalactions else 0
        legal_actions[3] = 1 if 'West' in legalactions else 0
        legal_actions[4] = 1 if 'Stop' in legalactions else 0

        self.buffer.append((state, action, reward, state_next, terminal,legal_actions))
       
        if len(self.buffer)==self.buffer_size:
            r = self.buffer.popleft()

    def storeCurrentState(self,state):
        self.states_t.append(state)
        if len(self.states_t)==self.buffer_size:
            r = self.states_t.popleft()

       
    
        

           
 
