import numpy as np
import random

class ReplayMemory(object):
    def __init__(self,state_size,buffer_size):

        self.states_t   = np.zeros((buffer_size,state_size[0],state_size[1]))
        self.states_t1   = np.zeros((buffer_size,state_size[0],state_size[1]))
        # self.states_t1  = np.zeros((state_size[0],state_size[1],buffer_size))
        self.actions    = np.zeros(buffer_size)
        self.rewards    = np.zeros(buffer_size)
        self.terminal   = np.zeros(buffer_size)
        self.next_idx   = 0
        self.idx_full   = 0
        self.size       = buffer_size
        self.full       = None
        self.last_included_item = 0
        self.legal_actions = np.zeros((buffer_size,5))        
                
    def sample(self, batch_size):
        assert batch_size<= self.idx_full
        indices        = np.random.choice(self.idx_full,batch_size)
        action_batch   = self.actions[indices]
        reward_batch   = self.rewards[indices]
        state_t_batch  = self.states_t[indices,:,:]
        state_t1_batch = self.states_t1[indices,:,:]
        terminal = self.terminal[indices]
        legal_actions = self.legal_actions[indices,:]
        # terminal = np.array([1.0 if self.terminal[idx] else 0.0 for idx in indices], dtype=np.float32)

        return state_t_batch, state_t1_batch, action_batch, legal_actions, reward_batch, terminal

    def store_state(self,state):
        if self.full is not None:
            self.next_idx = self.next_idx%self.size
            idx = self.next_idx
            self.next_idx = self.next_idx + 1
        else:
            idx = self.next_idx
            self.next_idx = self.next_idx + 1
            self.idx_full = self.idx_full + 1 
            if self.next_idx == self.size:
                self.full = 1
        self.states_t[idx,:,:] = state
        self.last_included_item = idx
        # print(idx)
        # return idx

    def store_state_random(self,state):
        if self.full is not None:
            idx = np.random.choice(self.size)
        else:
            idx = self.next_idx
            self.next_idx = self.next_idx + 1
            self.idx_full = self.idx_full + 1
            if self.next_idx == self.size:
                self.full = 1
        self.states_t[idx,:,:] = state
        self.last_included_item = idx
        # print(idx)
        # return idx


    def store_action_reward(self,action, reward, state, terminal):
        idx = self.last_included_item
        if action == 'North':
            self.actions[idx]=0
        elif action == 'East':
            self.actions[idx]=1
        elif action == 'South':
            self.actions[idx]=2
        elif action == 'West':
            self.actions[idx]=3
        elif action == 'Stop':
            self.actions[idx]=4
        
        self.rewards[idx]   = reward
        self.states_t1[idx,:,:] = state
        self.terminal[idx]       = terminal 

    def store_legal_actions(self, legal_actions):
        idx = self.last_included_item
        if 'North' in legal_actions:
            self.legal_actions[idx,0] = 1
        if 'East' in legal_actions:
            self.legal_actions[idx,1] = 1
        if 'South' in legal_actions:
            self.legal_actions[idx,2] = 1
        if 'West' in legal_actions:
            self.legal_actions[idx,3] = 1
        if 'Stop' in legal_actions:
            self.legal_actions[idx,4] = 1

           
 
