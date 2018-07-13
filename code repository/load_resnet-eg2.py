import numpy as np    
import tensorflow as tf  
slim = tf.contrib.slim  
import numpy as np  
import argparse  
import os  
from PIL import Image  
from datetime import datetime  
import math  
import time  
from resnet import *  
  
batch_size = 32  
height, width = 224, 224  
X = tf.placeholder(tf.float32, [None, height, width, 3])    
#Y = tf.placeholder(tf.float32, [None, 1000])    
#keep_prob = tf.placeholder(tf.float32) # dropout  
#keep_prob_fc = tf.placeholder(tf.float32) # dropout  
  
print ("-----------------------------main.py start--------------------------")  


def main():  
  
    # model  
    arg_scope = resnet_arg_scope()  
    with slim.arg_scope(arg_scope):  
        net, end_points = resnet_v2_101(X, is_training=True)  
  
    # initializer  
    #init = tf.global_variables_initializer()  
    sess = tf.Session()  
    #sess.run(init)   
      
    #reload model  
    saver1 = tf.train.Saver(tf.global_variables())  
    checkpoint_path = 'model/101/resnet_v2_101.ckpt'  
    saver1.restore(sess, checkpoint_path)  
      
    num_classes = 10  
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits2')  
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')  
    # initializer  
    init = tf.global_variables_initializer()  
    sess.run(init)   
      
    saver2 = tf.train.Saver(tf.global_variables())  
    #saver2.restore(sess, "model/101/fine-tune-1000")  
      
    # input  
    # input = X  
    # inputs = tf.random_uniform((batch_size, height, width, 3))  
    im = tf.read_file("m.jpg")  
    im = tf.image.decode_jpeg(im)  
    im = tf.image.resize_images(im, (width, height))  
    im = tf.reshape(im, [-1,height,width,3])  
    im = tf.cast(im, tf.float32)  
    inputs = im  
      
    # run  
    images = sess.run(inputs)  
    print (images)  
    start_time = time.time()  
    out_put = sess.run(net, feed_dict={X:images})  
    duration = time.time() - start_time  
    saver2.save(sess, "model/101/fine-tune", global_step=1000, write_meta_graph=False)  
      
    predict = tf.reshape(out_put, [-1, num_classes])  
    max_idx_p = tf.argmax(predict, 1)  
    print (out_put.shape)  
    print (sess.run(max_idx_p))  
    print ('run time:', duration)  
    sess.close()  
  
def test():  
  
    # model  
    arg_scope = resnet_arg_scope()  
    with slim.arg_scope(arg_scope):  
        net, end_points = resnet_v2_101(X, is_training=False)  
  
    # initializer  
    #init = tf.global_variables_initializer()  
    sess = tf.Session()  
    #sess.run(init)   
      
    #reload model  
    saver1 = tf.train.Saver(tf.global_variables())  
    checkpoint_path = 'model/101/resnet_v2_101.ckpt'  
    saver1.restore(sess, checkpoint_path)  
      
    num_classes = 10  
    net = slim.conv2d(net, num_classes, [1, 1], activation_fn=None, normalizer_fn=None, scope='logits2')  
    net = tf.squeeze(net, [1, 2], name='SpatialSqueeze')  
    # initializer  
    init = tf.global_variables_initializer()  
    sess.run(init)   
      
    saver2 = tf.train.Saver(tf.global_variables())  
    saver2.restore(sess, "model/101/fine-tune-1000")  
      
    # input  
    # input = X  
    # inputs = tf.random_uniform((batch_size, height, width, 3))  
    im = tf.read_file("m.jpg")  
    im = tf.image.decode_jpeg(im)  
    im = tf.image.resize_images(im, (width, height))  
    im = tf.reshape(im, [-1,height,width,3])  
    im = tf.cast(im, tf.float32)  
    inputs = im  
      
    # run  
    images = sess.run(inputs)  
    print (images)  
    start_time = time.time()  
    out_put = sess.run(net, feed_dict={X:images})  
      
    duration = time.time() - start_time  
    predict = tf.reshape(out_put, [-1, num_classes])  
    max_idx_p = tf.argmax(predict, 1)  
    print (out_put.shape)  
    print (sess.run(max_idx_p))  
    print ('run time:', duration)  
    sess.close()  
  
  
  
  
# main()  
test()  
