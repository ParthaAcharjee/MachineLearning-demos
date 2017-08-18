# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:23:57 2017

@author: ppa3551
"""

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

imgSize=28
inputs=imgSize*imgSize
outputs=10
channels=1 #Gray_Scale

with tf.name_scope('Dataset'):
    x=tf.placeholder(tf.float32,[None,inputs],name='Images')
    y=tf.placeholder(tf.float32,[None,outputs],name='Levels')
    
    #Reshape input image 1D array to 4D array for convulation input
    xImage=tf.reshape(x,[-1,imgSize,imgSize,channels],name='Resized')


def conv_layer(x,input_count,output_count,filter_width,strides=1,layer_name='Conv'):
    with tf.name_scope(layer_name): # Put each conv layer inside a name_scope
        shape=[filter_width,filter_width,input_count,output_count]
        w=tf.Variable(tf.truncated_normal(shape,stddev=0.05),name='Weights')
        b=tf.Variable(tf.constant(0.05,shape=[output_count]),name='Biases')
        
        filter_strides=[1,strides,strides,1]
        layer=tf.nn.conv2d(x,w,filter_strides,padding='SAME')+b
        
        # We can save computation by performing maxpooling before the ReLU
        return tf.nn.relu(layer)

def max_pooling(x,k=2,strides=2,layer_name='Pooling'):
    with tf.name_scope(layer_name):
        ksize=[1,k,k,1]
        stride=[1,strides,strides,1]
        layer=tf.nn.max_pool(x,ksize,stride,padding='SAME')
        
        return layer
    
def flatten(x, layer_name='Flat2Array'):
    with tf.name_scope(layer_name):
        tensorshape=x.get_shape()#[None, height, width, channel]
        elements=tensorshape[1:4].num_elements() 
        # flat to a 1D array
        return tf.reshape(x,[-1,elements]),elements


def fc_layer(x,inputs,outputs,layer_name='fc',relu=True):
    with tf.name_scope(layer_name):
        w=tf.Variable(tf.truncated_normal([inputs,outputs],stddev=0.05),name='Weights')
        b=tf.Variable(tf.constant(0.05,shape=[outputs]),name='Biases')
        
        layer=tf.matmul(x,w)+b
        if relu: return tf.nn.relu(layer)
        return layer

with tf.name_scope('Convolution_neural_network'):     
    Conv1=conv_layer(xImage,1,16,14,layer_name='Conv1')
    Pool1=max_pooling(Conv1,layer_name='Pool1')
    Conv2=conv_layer(Pool1,16,36,7,layer_name='Conv2')
    Pool2=max_pooling(Conv2,layer_name='Pool2')
    
    flat_input,fc1_inputs=flatten(Pool2,layer_name='Flatten')
    
    fc1=fc_layer(flat_input,fc1_inputs,128,layer_name='fc1')
    fc2=fc_layer(fc1,128,10,layer_name='fc2',relu=False)


with tf.name_scope('Optimizer'):
    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits=fc2,labels=y)
    cost=tf.reduce_mean(cross_entropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
    
with tf.name_scope('Performance'):
    y_pred=tf.nn.softmax(fc2)
    correct_pred=tf.equal(tf.argmax(y_pred,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
    
init = tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

batch_size=100
for itr in range(500):
    x_batch, y_batch = mnist.train.next_batch(batch_size)
    batch_dictionary={x:x_batch,y:y_batch}
    sess.run(optimizer,feed_dict=batch_dictionary)
    if itr%10==0: 
        acc=sess.run(accuracy,feed_dict=batch_dictionary)
        print('Iteration: {0} Training Accuracy: {1:.4%}'.format(itr+1,acc))
        
    