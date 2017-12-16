import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers

class neural_network:
    def __init__(self, height, width, discount,load_file,lr):
        self.load_file = load_file
        self.lr = lr
        self.network_name = 'qnet'
        self.sess = tf.Session()
        self.x = tf.placeholder('float', [None, height,width,1],name=self.network_name + '_x')
        self.actions = tf.placeholder("float", [None, 5], name=self.network_name + '_actions')
        self.rewards = tf.placeholder("float", [None], name=self.network_name + '_rewards')
        self.terminals = tf.placeholder("float", [None], name=self.network_name + '_terminals')
        self.legal  = tf.placeholder("float", [None, 5], name=self.network_name + '_legal_actions')
        self.q_t = tf.placeholder('float', [None, 5], name=self.network_name + '_q_t')
        self.counter = 0
        self.frequency = 10;

        num_actions = 5
        num_channels = 1
        #input_layer = tf.reshape(state,[-1,self.n_features[0],self.n_features[1],1])
        with tf.variable_scope('train_q'):
            conv1 = tf.contrib.layers.conv2d(
                inputs= self.x,
                num_outputs = 8,
                kernel_size = 3,
                padding = "SAME",
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())
            conv2 = tf.contrib.layers.conv2d(
                inputs= conv1,
                num_outputs = 16,
                kernel_size = 3,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                padding = "SAME")
            conv3 = tf.contrib.layers.conv2d(
                 inputs= conv2,
                 num_outputs = 32,
                 kernel_size = 3,
                 weights_initializer=tf.contrib.layers.xavier_initializer(),
                 biases_initializer=tf.zeros_initializer(),
                 padding = "SAME")
            flatten = tf.contrib.layers.flatten(inputs = conv3)
            dense1 = tf.contrib.layers.fully_connected(
                inputs = flatten,
                num_outputs = 256, # number of neurons
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn = tf.nn.relu)
            self.y = tf.contrib.layers.fully_connected(
                inputs = dense1,
                num_outputs = 5,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn = None)

        with tf.variable_scope('target_q'):
            conv1 = tf.contrib.layers.conv2d(
                inputs= self.x,
                num_outputs = 8,
                kernel_size = 3,
                padding = "SAME",
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer())

            conv2 = tf.contrib.layers.conv2d(
                inputs= conv1,
                num_outputs = 16,
                kernel_size = 3,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                padding = "SAME")
            conv2 = tf.contrib.layers.conv2d(
                inputs= conv2,
                num_outputs = 32,
                kernel_size = 3,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                padding = "SAME")
            flatten = tf.contrib.layers.flatten(inputs = conv3)
            dense1 = tf.contrib.layers.fully_connected(
                inputs = flatten,
                num_outputs = 256, # number of neurons
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn = tf.nn.relu)
            self.q_next = tf.contrib.layers.fully_connected(
                inputs = dense1,
                num_outputs = 5,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                biases_initializer=tf.zeros_initializer(),
                activation_fn = None)


        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_q')
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
      
        self.discount = tf.constant(discount)
        self.yj = tf.add(self.rewards, tf.multiply(1.0-self.terminals, tf.multiply(self.discount,\
          -1.1*tf.abs(tf.reduce_max(self.q_t))+ tf.reduce_max(tf.multiply\
            (self.q_t+1.1*tf.abs(tf.reduce_max(self.q_t)),self.legal),reduction_indices=1))))
        self.Q_pred = tf.reduce_sum(tf.multiply(self.y,self.actions), reduction_indices = 1)
        self.loss = tf.reduce_sum(tf.pow(tf.subtract(self.yj, self.Q_pred), 2))
        
        self.optimizer = tf.train.AdamOptimizer(self.lr)
        self.grads, self.variables = zip(*self.optimizer.compute_gradients(self.loss,
                                                            var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'train_q')))
        self.grads, _ = tf.clip_by_global_norm(self.grads, 20)
        self.grad_norm = tf.global_norm(self.grads)
        self.train_op = self.optimizer.apply_gradients(zip(self.grads, self.variables))

        self.saver = tf.train.Saver(max_to_keep=0)

        self.sess.run(tf.global_variables_initializer())

        if self.load_file is not None:
            print('Loading checkpoint...')
            self.saver.restore(self.sess,self.load_file)

        
    def train(self,bat_s,bat_a,bat_t,bat_n,bat_r,batch_legal):
        feed_dict={self.x: bat_n, self.q_t: np.zeros((bat_n.shape[0],5)), self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r, self.legal:batch_legal}
        q_t = self.sess.run(self.q_next,feed_dict=feed_dict)
       
        self.counter = self.counter + 1;
        
        feed_dict={self.x: bat_s, self.q_t: q_t, self.actions: bat_a, self.terminals:bat_t, self.rewards: bat_r, self.legal:batch_legal}
        cost, grad_norm_eval, _ = self.sess.run([self.loss, self.grad_norm,self.train_op], feed_dict=feed_dict)
        if (self.counter%self.frequency == 0):
            self.sess.run(self.target_replace_op)
        return cost

    def save_ckpt(self,filename):
        self.saver.save(self.sess, filename)

