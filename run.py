import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn
import tensorflow as tf


from policyGradient import observation, target, action, \
action_prob, train_value, train_action #gradient

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

env = gym.make('CartPole-v0')
# logDir = 'tmp/cartpole-experiment-1'
# if os.path.isdir(logDir):
#     shutil.rmtree(logDir)
# env = wrappers.Monitor(env, logDir)

gamma = 0.9
batch_size = 10
keep_size = 20

for i_episode in range(500):
    obs = env.reset()
    memory = []
    for t in range(500):
        #env.render()
        #print(observation)
        action_p = sess.run(action_prob, feed_dict={observation:[obs]})
        a = 0 if action_p > np.random.rand() else 1
        obs, reward, done, info = env.step(a)
        memory.append([obs,a,reward])
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    for i, unit in enumerate(memory):
        G = np.sum([gamma**j*item[2] for j,item in enumerate(memory[i:])])
        unit.append(G)

    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    for chunk in qn.chunks(memory,batch_size):
        obsChunk, actionChunk, _, targetChunk = list(zip(*chunk))
        obsChunk = np.stack(obsChunk)
        targetChunk = [[x] for x in targetChunk]
        actionChunk = [[1,0] if x==0 else [0,1] for x in actionChunk]
        sess.run(train_action,feed_dict={observation:obsChunk,
        action:actionChunk, target:targetChunk})
        # sess.run(gradient,feed_dict={observation:obsChunk,
        # action:actionChunk, target:targetChunk})
        sess.run(train_value,feed_dict={observation:obsChunk,
        target:targetChunk})
