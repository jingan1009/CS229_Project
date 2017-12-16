
from __future__ import print_function

import tensorflow as tf
import numpy
#import matplotlib.pyplot as plt
rng = numpy.random

# Parameters
learning_rate = 0.1
training_epochs = 20000
display_step = 1000


train_X = numpy.loadtxt('fdata.txt',delimiter=',')
train_Y =  numpy.loadtxt('qdata.txt')

n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Set model weights
W1 = tf.Variable(rng.randn(), name="weight")
W2 = tf.Variable(rng.randn(), name="weight")
W3 = tf.Variable(rng.randn(), name="weight")
W0 = tf.Variable(rng.randn(), name="weight")
#b = tf.Variable(rng.randn(), name="bias")

# Construct a linear model
pred = tf.add(X[:,0]*W0 + X[:,1]*W1 + X[:,2]*W2 , X[:,3]*W3)

# Mean squared error
cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)
optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate,l2_regularization_strength=0.0001).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        sess.run(optimizer, feed_dict={X: train_X, Y:train_Y })

        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), \
                "W0=", sess.run(W0),"W1=", sess.run(W1), \
                "W2=", sess.run(W2),"W3=", sess.run(W3))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W0=", sess.run(W0),"W1=", sess.run(W1), \
                "W2=", sess.run(W2),"W3=", sess.run(W3), '\n')

