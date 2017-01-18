# import tensorflow as tf
#
# observation = tf.placeholder(tf.float32, [None,2], 'observation')
# target = tf.placeholder(tf.float32, [None,1], 'observation')
#
# w_0 = tf.Variable(tf.truncated_normal([2,6], stddev=0.1),
# name='w_v1')
# b_0 = tf.Variable(tf.constant(0.1, shape=[6]), name='b_v1')
# pa_0 = tf.matmul(observation, w_v1) + b_v1
# mse = tf.contrib.losses.mean_squared_error(pa_0, target)
#
# train_value = tf.train.AdamOptimizer(learning_rate=value_lr)\
# .minimize(mse)

from sklearn.linear_model import ElasticNet
import pickle
import numpy as np

with open('data/mountain-car-1.pkl','rb') as mf:
    total = pickle.load(mf)

keep_size = 1000
gamma = 0.98
x = [[],[],[]]
y = [[],[],[]]
for memory in total:
    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    for unit in memory:
        unit[2] = 0
    memory[-1][2] = 1

    for i, unit in enumerate(memory):
        G = np.sum([gamma**j*item[2] for j,item in enumerate(memory[i:])])
        x[unit[1]].append(unit[0])
        y[unit[1]].append(G)



models = [ElasticNet(),ElasticNet(),ElasticNet()]
for i, enet in enumerate(models):
    enet.fit(x[i],y[i])
    
with open('models/mountain-car-simple.pkl','wb') as mf:
    pickle.dump(models,mf)
