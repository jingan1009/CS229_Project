from NeuralNetwork import *
from pacman import Directions
from replayMemoryQueue import *
from game import Directions, Agent, Actions
from collections import deque

import game
import numpy as np
import random,util,time
import random
import util
import sys
import time
import tensorflow as tf

class DeepQAgent(Agent):
    def __init__(self, numTraining=100, width= 10, height=10):

        print("Initializing DQN Agent")

        self.load_file= None
        self.save_file= None
        self.save_interval= 10000

        # Training parameters
        self.train_start= 5000    # Episodes before training starts
        self.batch_size= 32       # Batch size
        self.mem_size= 50000      # Replay memory size
        self.train_freq = 1
        self.width = width
        self.height = height
        self.num_training = numTraining
        self.discount= 0.99       # Discount rate (gamma value)
        self.lr = 0.00025         # Learning rate
        self.lr_end= 0.0005       # Learning rate end value
        self.lr_steps= 250000

        # Epsilon-greedy
        self.eps= 1.0             # Epsilon start value
        self.eps_final= 0.1       # Epsilon end value
        self.eps_step= 10000      # Epsilon steps between start and end (linear)

        # Tensorflow
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        self.sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        self.qnet = neural_network(self.height, self.width, self.discount, self.load_file,self.lr)

        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        # Q and cost
        self.Q_global = []
        self.cost_disp = 0     

        # Stats
        self.cnt = 0#self.qnet.sess.run(self.qnet.global_step)
        self.local_cnt = 0

        self.numeps = 0
        self.last_score = 0
        self.s = time.time()
        self.last_reward = 0.
        self.is_test = False
        self.testcounter = 0
        self.accumTestRewards = 0
        self.eps_store=self.eps

        self.replay_mem = ReplayMemoryQueue(self.mem_size)
        self.last_scores = deque()

    def get_direction(self, value):
        if value == 0.:
            return Directions.NORTH
        elif value == 1.:
            return Directions.EAST
        elif value == 2.:
            return Directions.SOUTH
        elif value == 3.:
            return Directions.WEST
        else:
            return Directions.STOP


    def get_action(self, direction):
        if direction == Directions.NORTH:
            return 0.
        elif direction == Directions.EAST:
            return 1.
        elif direction == Directions.SOUTH:
            return 2.
        elif direction == Directions.WEST:
            return 3.
        else:
            return 4.

    def observation_step(self, state):
        if self.last_action is not None:
            # current experience state
            self.last_state = np.copy(self.current_state)
            self.current_state = self.drawStates(state)

            # current experience reward
            self.current_score = state.getScore()
            reward = self.current_score - self.last_score
            self.last_score = self.current_score

            if reward > 20:
                self.last_reward = 200.   # Eat ghost
            elif reward > 0:
                self.last_reward = 50.    # Eat food
            elif reward < -10:
                self.last_reward = -500.  # Lost game 
                self.won = False
            elif reward < 0:
                self.last_reward = -1.    # Delay time

            
            if(self.terminal and self.won):
                self.last_reward = 500.   # Win game
            self.ep_rew += self.last_reward

            # Store last experience tuple into memory 
            legal = state.getLegalActions(0)
            self.legal_actions = self.get_legal_action_onehot(legal)
            terminal = self.terminal
            self.replay_mem.storeAll(self.last_state, self.last_action,float(self.last_reward), self.current_state, terminal,self.legal_actions)

            # Save model if we want to
            if(self.save_file):
                if self.local_cnt > self.train_start and self.local_cnt % self.save_interval == 0:
                    self.qnet.save_ckpt('saves/model-' + self.save_file + "_" + str(self.cnt) + '_' + str(self.numeps))
                    print('Model saved')

            # Train the pacman agent
            if not self.is_test and self.cnt%self.train_freq==0:
                self.cnt = self.cnt+1
                self.train()
                self.eps = max(self.eps_final,
                                 1.00 - float(self.cnt)/ float(self.eps_step))
                self.lr = max(self.lr_end,
                                 self.lr - self.lr_end*float(self.cnt)/ float(self.lr_steps))

        # Next frame
        self.local_cnt += 1
        self.frame += 1
        


    def observationFunction(self, state):
        # Do observation
        self.terminal = False
        self.observation_step(state)
        #print(self.params['eps'])
        return state

    def final(self, state):
        # Next
        self.ep_rew += self.last_reward

        # Do observation
        self.terminal = True
        self.observation_step(state)

        # Test after every 100 games after the first 1000 training games
        NUM_UPDATE = 50
        if self.numeps>399 and (self.numeps-400)% NUM_UPDATE==0:
            self.is_test = True # Start testing 
            self.testcounter += 1

            if self.testcounter==1:
                self.eps_store = self.eps
            # self.lr_store = self.params['lr']
            # turn off epsilon during testing
            self.eps = 0
            # self.params['lr'] = 0
            print('Going over test set')
            self.accumTestRewards += self.ep_rew

        # Print stats
        log_file = open('./logs/'+str(self.general_record_time)+'-l-'+str(self.width)+'-m-'+str(self.height)+'-x-'+str(self.num_training)+'.log','a')
        log_file.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f \n" %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.eps))
        # log_file.write("| Q: %10f | won: %r \n" % ((np.amax(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.write("# %4d | steps: %5d | steps_t: %5d | t: %4f | r: %12f | e: %10f \n" %
                         (self.numeps,self.local_cnt, self.cnt, time.time()-self.s, self.ep_rew, self.eps))
        # sys.stdout.write("| Q: %10f | won: %r \n" % ((np.amax(self.Q_global, default=float('nan')), self.won)))
        sys.stdout.flush()

    def train(self):
        # Train
        batch_s, batch_a, batch_t, batch_n, batch_r, batch_legal = self.replay_mem.sample(self.batch_size)
        batch_a =  self.get_onehot(np.array(batch_a))
        if (self.local_cnt > self.train_start):
            self.cost_disp = self.qnet.train(batch_s, batch_a, batch_t, batch_n, batch_r, batch_legal)
        if not self.is_test: 
            train_file = open('./trainLoss/'+str(self.general_record_time)+'-l-'+str(self.width)+'-m-'+str(self.height)+'-x-'+str(self.num_training)+'.log','a')
            train_file.write("# %4d | loss: %12f \n" %
                         (self.numeps,self.cost_disp))
        


    def get_onehot(self, actions):
        """ Create list of vectors with 1 values at index of action in list """
        actions_onehot = np.zeros((self.batch_size, 5))
        for i in range(len(actions)):                                           
            actions_onehot[i][int(actions[i])] = 1      
        return actions_onehot   

    def get_legal_action_onehot(self,legal_actions):
        legalactions_onehot = np.zeros(5)
        if 'North' in legal_actions:
            legalactions_onehot[0] = 1 
        if 'East' in legal_actions:
            legalactions_onehot[1] = 1 
        if 'South' in legal_actions:
            legalactions_onehot[2] = 1 
        if 'West' in legal_actions:
            legalactions_onehot[3] = 1 
        if 'Stop' in legal_actions:
            legalactions_onehot[4] = 1 
        return legalactions_onehot

    def getAction(self, state):
        legal = state.getLegalActions(0)
        legal_actions = self.get_legal_action_onehot(legal)
        if np.random.rand() > self.eps:

            self.Q_c = self.qnet.sess.run(
                self.qnet.y,
                feed_dict = {self.qnet.x: np.reshape(self.current_state,
                             (1, self.height, self.width, 1)),
                             self.qnet.q_t: np.zeros((1,5)),
                             self.qnet.actions: np.zeros((1, 5)),
                             self.qnet.terminals: np.zeros(1),
                             self.qnet.rewards: np.zeros(1)})[0]
            self.Q_global.append(max(self.Q_c))
            self.Q_c = legal_actions*(self.Q_c+1.1*np.max(abs(self.Q_c)))
            arg_action = np.argwhere(self.Q_c == np.amax(self.Q_c))
            if len(arg_action) > 1:
                action = self.get_direction(
                    arg_action[np.random.randint(0, len(arg_action))][0])
            else:
                action = self.get_direction(arg_action[0][0])
        else:
            while True:
                action = self.get_direction(np.random.randint(0, 5))
                if action in legal:
                    break
        self.last_action = self.get_action(action)
        return action

    def drawStates(self, state):
        width = state.data.layout.width
        height = state.data.layout.height
        stateMatrix = np.zeros((height,width))
        wcolour = 1
        gcolour = 43
        agcolour = 86
        fcolour = 129
        ccolour = 172
        pcolour = 215

        def drawWalls(state, state_matrix, wcolour):
            walls = state.data.layout.walls
            width = state.data.layout.width
            height = state.data.layout.height
            for x in range(width):
                for y in range(height):
                    state_matrix[-1-y][x] = wcolour if walls[x][y] else 0
            return state_matrix

        def drawFood(state, state_matrix, fcolour):
            food = state.data.layout.food
            width = state.data.layout.width
            height = state.data.layout.height
            for x in range(width):
                for y in range(height):
                    if food[x][y]:
                        state_matrix[-1-y][x] = fcolour
            return state_matrix


        def drawCapsules(state, state_matrix, ccolour):
            capsules = state.data.layout.capsules
            width = state.data.layout.width
            height = state.data.layout.height
            for x,y in capsules:
                x = int(x)
                y = int(y)
                state_matrix[-1-y][x] = ccolour
            return state_matrix

        def drawGhosts(state, state_matrix, gcolour):
            AgentStates = state.data.agentStates
            for s in AgentStates:
                if not s.isPacman:
                    if s.scaredTimer==0:
                        x,y = s.configuration.getPosition()
                        x = int(x)
                        y = int(y)
                        state_matrix[-1-y][x] = gcolour
            return state_matrix

        def drawScaredGhosts(state, state_matrix, agcolour):
            AgentStates = state.data.agentStates
            for s in AgentStates:
                if not s.isPacman:
                    if s.scaredTimer>0:
                        x,y = s.configuration.getPosition()
                        x = int(x)
                        y = int(y)
                        state_matrix[-1-y][x] = agcolour
            return state_matrix

        def drawPacman(state, state_matrix, pcolour):
            AgentStates = state.data.agentStates
            for s in AgentStates:
                if s.isPacman:
                    x,y = s.configuration.getPosition()
                    x = int(x)
                    y = int(y)
                    state_matrix[-1-y][x] = pcolour
            return state_matrix

        stateMatrix = drawWalls(state, stateMatrix, wcolour)
        stateMatrix = drawGhosts(state, stateMatrix, gcolour)
        stateMatrix = drawScaredGhosts(state, stateMatrix, agcolour)
        stateMatrix = drawPacman(state, stateMatrix, pcolour)
        stateMatrix = drawFood(state, stateMatrix, fcolour)
        stateMatrix = drawCapsules(state, stateMatrix, ccolour)
        stateMatrix_new = np.expand_dims(stateMatrix, axis=2)
        return stateMatrix_new

    def registerInitialState(self, state): # inspects the starting state

        # Reset reward
        self.last_score = 0
        self.current_score = 0
        self.last_reward = 0.
        self.ep_rew = 0

        # Reset state
        self.last_state = None
        self.current_state = self.drawStates(state)

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []
        self.delay = 0

        # Next
        self.frame = 0

        if self.testcounter==50:
            test_avg = self.accumTestRewards/50
            self.saveResults(test_avg)
            self.testcounter=0
            self.accumTestRewards =0 
            self.is_test = False
            self.eps = self.eps_store

        if not self.is_test:
            self.numeps += 1
            
    def saveResults(self,test_avg):
        results_file = open('./testRewards/'+str(self.general_record_time)+'-l-'+str(self.width)+'-m-'+str(self.height)+'-x-'+str(self.num_training)+'.log','a')
        results_file.write("# %4d | num_test: 100 | Average test reward: %12f\n" %(self.numeps,test_avg))
        
