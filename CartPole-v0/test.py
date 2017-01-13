#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:23:33 2017

@author: qn
"""

# x = [[i] for i in range(5)]
#
# for item in x:
#     item.append(item[0]+1)

import tensorflow as tf

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

a = tf.constant([[1],[2],[3]])
b = tf.constant([[1,2,3],[4,5,6],[9,8,7]])

d1 = tf.constant([[1],[3],[5]])
d2 = tf.constant([[2],[4],[6]])
d3 = tf.concat(1,[d1,d2])
d5 = tf.constant([[1,0],[0,1],[1,0]])
d6 = tf.multiply(d3,d5)
d7 = tf.reduce_sum(d6,axis=1)

c = tf.multiply(b,a)

c_r = sess.run(c)
dr = sess.run(d3)
dr2 = sess.run(d7)

