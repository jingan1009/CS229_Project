# qlearningAgents.py
# ------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *
#from replaymemory import *
from replayMemoryQueue import *
import random,util,math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers 

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)

        "*** YOUR CODE HERE ***"
        self.Q = util.Counter() 


    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.Q[(state,action)];

        util.raiseNotDefined()

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"

        legalActions = self.getLegalActions(state)
        if len(legalActions)==0:
          return None

        max_q = -100000
        max_a = None

        for x in legalActions:
          if self.getQValue(state,x)>=max_q: #or max_a==None:
            max_q = self.getQValue(state,x)
            max_a = x
        return max_a;

        util.raiseNotDefined()

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        action = self.computeActionFromQValues(state)

        if action==None:
          return 0.0
        else:
          return self.getQValue(state,self.computeActionFromQValues(state))

        util.raiseNotDefined()

   

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legalActions = self.getLegalActions(state)


        "*** YOUR CODE HERE ***"
        if len(legalActions)==0:
          return None
          
        prob = util.flipCoin(self.epsilon)
       
        # print(self.epsilon)
        if prob==1:
          # print('random choice')
          return random.choice(legalActions)
        else:
          return self.computeActionFromQValues(state)


    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"

        # Update Q(state,action)
        learning_rate = self.alpha
        discount = self.discount

        max_q=self.computeValueFromQValues(nextState)

        # for x in legalActions:
        #   if self.getQValue(nextstate,x)>=max_q:
        #     max_q = self.getQValue(nextState,x)        
        self.Q[(state,action)] = self.getQValue(state,action) + learning_rate*(reward+discount*max_q-self.getQValue(state,action))

        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0,**args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)
        # self.replayMem = replay # A replayMemory object

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights

    def getFeatures(self,state,action):
        return self.featExtractor.getFeatures(state,action)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "* YOUR CODE HERE *"
        sum_q = 0
        feats = self.featExtractor.getFeatures(state,action)
        for i in feats:
          sum_q = sum_q + feats[i]*self.weights[i]

        # self.Q[(state,action)]
        return sum_q

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "* YOUR CODE HERE *"
        max_q=self.computeValueFromQValues(nextState)
        feats = self.featExtractor.getFeatures(state,action)

        difference = reward+self.discount*max_q-self.getQValue(state,action)

        for i in feats:
          self.weights[i] = self.weights[i]+self.alpha*(difference)*feats[i]
        #print(self.weights[i])

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            "* YOUR CODE HERE *"
            # print(self.weights)
            pass

class SimpleNeuralNetAgent(ApproximateQAgent):
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        # self.weights = {'closest-food': -2.262592596651226, 'bias': 204.35210901295517, \
        # '#-of-ghosts-1-step-away': -154.98480951737852, 'eats-food': 282.4062467344489}
        # self.weights = {'closest-food': -0.852518, 'bias': 205.99, \
        # '#-of-ghosts-1-step-away': -16.3985, 'eats-food': 276.808}
        
        # For 5 test games
        # self.weights = {'closest-food': 0.93643, 'bias': 266.787, \
        # '#-of-ghosts-1-step-away': -0.400376, 'eats-food': 266.972}
        
        # For 50 test games
        # self.weights = {'closest-food': 0.594164, 'bias': 216.596, \
        # '#-of-ghosts-1-step-away': 0.441064, 'eats-food': 252.362}

        # 50 games and different optimizer
        self.weights = {'closest-food': -0.523033, 'bias': 204.692, \
        '#-of-ghosts-1-step-away': -0.83809, 'eats-food': 259.864}

    # def update(self, state, action, nextState, reward):
    #     pass


    def getWeight(self):
        return self.weights

    def getFeatures(self,state,action):
        return self.featExtractor.getFeatures(state,action)

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        
        feats = self.featExtractor.getFeatures(state,action)
        # print(feats)
        sum_q=0
        for idx,val in enumerate(feats):
          # print(idx)
          sum_q = sum_q + feats[val]*self.weights[val]

        # self.Q[(state,action)]
        return sum_q



class DeepQNetAgent(PacmanQAgent):
  
    def __init__(self, n_features=[7,7], n_actions=5, alpha=0.0002,
        gamma=0.95, epsilon=1, learning_freq=1, replace_target_iter=100,
        buffer_size=100000, batch_size=32, e_greedy_increment=None,
        output_graph=False, **args):
        PacmanQAgent.__init__(self,**args)
        
        self.n_actions = n_actions
        self.n_features = n_features
        
        self.replace_target_iter = replace_target_iter
        self.learning_freq = learning_freq
        self.buffer_size = buffer_size
        self.batch_size = batch_size
       
        self.is_dqn = 1
        self.clip_val = 10
        
        self.replay_memory = ReplayMemoryQueue(self.buffer_size)

        self.add_placeholders_op()
        self.q = self.q_values(self.s, scope="q_eval", reuse=False)
        self.target_q = self.q_values(self.s_t, scope="q_target", reuse=False)

        # Add the operator to update target network
        self.add_update_target('q_eval','q_target')
        # Add loss and optimizer
        self.add_loss(self.q,self.target_q)
        self.add_optimizer('q_eval')

        # Create tensorflow session
        self.sess = tf.Session()

            # @@@@ ADD SUMMARY IF WE WANT @@@@@ 

            # Initialize variables
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.update_target_op)
            
            # Save networks weights
        self.saver = tf.train.Saver()

        

    def update_target_params(self):
        """
        Update parametes of Q' with parameters of Q
        """
        self.sess.run(self.update_target_op)


    def add_placeholders_op(self):
        #state_shape = list(self.env.observation_space.shape)
        self.state_shape = (self.n_features[0], self.n_features[1])
        
        self.s = tf.placeholder(tf.float32, shape=(None,) + self.state_shape)
        self.a = tf.placeholder(tf.int32, shape=(None,))
        self.legal_a = tf.placeholder(tf.float32, shape=(None,)+ (5,))
        self.r = tf.placeholder(tf.float32, shape=(None,))
        self.s_t = tf.placeholder(tf.float32, shape=(None,) + self.state_shape)
        # self.terminate = tf.placeholder(tf.float32, shape=(None,))
        self.lr = tf.placeholder(tf.float32, shape=())

    def getQValue(self,state,action):
        state_im = self.replay_memory.states_t[-1]
        state_im = np.reshape(state_im,(1,self.n_features[0],self.n_features[1]))

        q_values = self.sess.run([self.q],feed_dict={self.s: state_im})
        if action == 'North':
            return q_values[0][0][0]
        elif action == 'East':
            return q_values[0][0][1]
        elif action == 'South':
            return q_values[0][0][2]
        elif action == 'West':
            return q_values[0][0][3]
        elif action == 'Stop':
            return q_values[0][0][4]


    def q_values(self,state,scope,reuse=False):

        num_actions = 5
        # input_layer = state
        num_channels = 1
        with tf.variable_scope(scope,reuse=reuse):

            input_layer = tf.reshape(state,[-1,self.n_features[0],self.n_features[1],1])
            # print(input_layer.shape)
            conv1 = tf.contrib.layers.conv2d(
                inputs= input_layer,
                num_outputs = 16,
                kernel_size = 3,
                padding = "SAME")
            

  # Pooling layer  # Max pooling layer # output = tensor of shape [batch_size,img_width/2,img_height/2,filters]
            # pool1 = tf.contrib.layers.max_pool2d(
            #     inputs = conv1,
            #     kernel_size = [2,2],
            #     stride = 2)

            conv2 = tf.contrib.layers.conv2d(
                inputs= conv1,
                num_outputs = 32,
                kernel_size = 3,
                padding = "SAME")
            

  # Pooling layer  # Max pooling layer # output = tensor of shape [batch_size,img_width/2,img_height/2,filters]
            # pool2 = tf.contrib.layers.max_pool2d(
            #     inputs = conv2,
            #     kernel_size = [2,2],
            #     stride = 2)

  #           conv3 = tf.contrib.layers.conv2d(
  #               inputs= pool2,
  #               num_outputs = 32,
  #               kernel_size = 3,
  #               padding = "SAME")
            

  # # Pooling layer  # Max pooling layer # output = tensor of shape [batch_size,img_width/2,img_height/2,filters]
  #           pool3 = tf.contrib.layers.max_pool2d(
  #               inputs = conv3,
  #               kernel_size = [2,2],
  #               stride = 2)           
            

            flatten = tf.contrib.layers.flatten(inputs = conv2)
            

  # Fully connected layer with 200 neurons
            dense1 = tf.contrib.layers.fully_connected(
                inputs = flatten,
                num_outputs = 256, # number of neurons
                activation_fn = tf.nn.relu)
            # print(dense1.shape)

  # Final output layer # Outputs q-values of 5 actions: left,right,up,down,stop
            output = tf.contrib.layers.fully_connected(
                inputs = dense1,
                num_outputs = 5)
            # print(output.shape)
        return output


    def add_update_target(self, train_scope, target_scope):
        # for trainable network 
        train = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=train_scope), key= lambda x:x.name) 

        # for target network
        target = sorted(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=target_scope), key= lambda x:x.name) 
        
        assignments = [tf.assign(ta, tr) for tr, ta in zip(train, target)]
        self.update_target_op = tf.group(*assignments)

    def train_network(self):
        state_t_batch, state_t1_batch, action_batch, legal_action_batch,reward_batch= self.replay_memory.sample(self.batch_size)
        # @@@@@@@ IMPORTANT ACTION_BATCH SHOULD BE STORED AS INTEGER
        

        # train eval network
        fd = {
            # inputs
            self.s: np.array(state_t_batch),
            self.a: np.array(action_batch),
            self.legal_a: np.array(legal_action_batch),
            self.r: np.array(reward_batch),
            self.s_t: np.array(state_t1_batch), 
            # self.terminate: terminate,
            self.lr: self.alpha}

        loss_eval, grad_norm_eval,_ = self.sess.run([self.loss, self.grad_norm, 
                                                 self.train_op], feed_dict=fd)
        # print(loss_eval)


        save_path = self.saver.save(self.sess, 'weights/smallGrid1.ckpt')
        # print("Model saved in file: %s" % save_path)

        
        self.epsilon = self.epsilon - 0.00009 if self.epsilon > 0.1 else 0.1
        self.alpha = self.alpha - 0.0000002 if self.alpha > 0.0001 else 0.0001

    def add_loss(self,q,q_next):
        num_actions = 5
        Q_samp = self.r + self.discount * tf.reduce_max(tf.multiply(self.legal_a,q_next), axis=1)       
        # Q_samp = self.r + (1 - self.terminate) * self.gamma * tf.reduce_max(tf.multiply(self.legal_a,q_next), axis=1)       
        # print(self.sess.run(self.legal_a))
        # print(tf.one_hot(self.a,num_actions))
        Q = tf.diag_part(tf.matmul(q, tf.transpose(tf.one_hot(self.a, num_actions))))


        # Q = tf.diag_part(tf.matmul(q, tf.transpose(tf.one_hot(self.a, num_actions))))
        self.loss = tf.reduce_mean((Q_samp - Q)**2)
        
    def add_optimizer(self,scope):
        optimizer = tf.train.AdamOptimizer(self.lr)
        grads, variables = zip(*optimizer.compute_gradients(self.loss,
                                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)))        
        # if self.config.grad_clip:
        grads, _ = tf.clip_by_global_norm(grads, self.clip_val)
        self.grad_norm = tf.global_norm(grads)
        self.train_op = optimizer.apply_gradients(zip(grads, variables))

    def update(self, state, action, nextState, reward):
        pass

    def drawStates(self,state):
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
            # print('walls')
            # print(state_matrix)

            return state_matrix

        def drawFood(state, state_matrix, fcolour):
            food = state.data.layout.food
            width = state.data.layout.width
            height = state.data.layout.height

            for x in range(width):
                for y in range(height):
                    if food[x][y]:
                        state_matrix[-1-y][x] = fcolour 
            # print('food')

            # print(state_matrix)

            return state_matrix


        def drawCapsules(state, state_matrix, ccolour):
            capsules = state.data.layout.capsules
            width = state.data.layout.width
            height = state.data.layout.height

            for x,y in capsules:
                x = int(x)
                y = int(y)
                state_matrix[-1-y][x] = ccolour 
            # print('caps')

            # print(state_matrix)
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
            # print('ghosts')
            # print(state_matrix)
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
            # print('scared ghosts')
            # print(state_matrix)
            return state_matrix


        def drawPacman(state, state_matrix, pcolour):
            AgentStates = state.data.agentStates
            for s in AgentStates:
                if s.isPacman:
                    x,y = s.configuration.getPosition()
                    x = int(x)
                    y = int(y)
                    state_matrix[-1-y][x] = pcolour
            # print('pacman')
            # print(state_matrix)
            return state_matrix

        
        stateMatrix = drawWalls(state, stateMatrix, wcolour)
        stateMatrix = drawFood(state, stateMatrix, fcolour)
        stateMatrix = drawCapsules(state, stateMatrix, ccolour)
        stateMatrix = drawScaredGhosts(state, stateMatrix, agcolour)
        stateMatrix = drawPacman(state, stateMatrix, pcolour)
        stateMatrix = drawGhosts(state, stateMatrix, gcolour)
        
        
        
       

        return stateMatrix





