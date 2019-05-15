# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 17:56:51 2019

@author: amir

LINEAR REGRESSION WITH ONE DIMENSIONAL INPUT EXAMPLE IN TENSORFLOW

"""

import tensorflow as tf
import numpy as np

# Input, output
x = tf.placeholder(name='x', shape=[4,1], dtype=tf.float32)
y = tf.placeholder(name='y', shape=[1], dtype=tf.float32)

# Model params
w = tf.get_variable(name='weight', shape=[1,4], initializer=tf.initializers.truncated_normal())
b = tf.get_variable(name='bias', initializer=tf.zeros([1])) #If initializer has shape, then 'shape=' is not needed

# Model
#y_hat = w * x + b # in case of only one input
y_hat = tf.squeeze(tf.matmul(w, x)) + b
print(y_hat.get_shape().as_list())
loss = tf.losses.mean_squared_error(labels=y, predictions=y_hat)

# Training op
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_op  = optimizer.minimize(loss)

sess = tf.Session()

# Initialize all variables
init_op = tf.global_variables_initializer()
sess.run(init_op)

for i in range(2000):
    
    x_np = np.random.random(4)
    x_np = np.reshape(x_np, [4, 1])
    
    y_np = x_np[0] * 2 + x_np[1] * 3 + x_np[2] * 4 + x_np[3] * 5 + 6 + np.random.random()
    y_np = np.reshape(y_np, [1])
    
    w_, b_, _, loss_ = sess.run([w, b, train_op, loss], feed_dict={x: x_np, y: y_np})
    
    if i % 10 == 0:
        print('W: ' + str(np.squeeze(w_)) + ' b:' + str(b_) + ' loss:' + str(loss_))
        