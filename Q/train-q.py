import numpy as np
import tensorflow as tf
#import Q as q
import Q2 as q
import qn

import gym
from gym import wrappers
import shutil
import os

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver.restore(sess, '../models/moutainCar-pretrain-q2')

env = gym.make('MountainCar-v0')
# logDir = 'tmp/MountainCar-v0-mc'
# if os.path.isdir(logDir):
#     shutil.rmtree(logDir)
# env = wrappers.Monitor(env, logDir)

keep_size = 2000
#gamma = 0.99
epsilonUp = 50
epsilonDn = 100
batch_size = 50

for i_episode in range(int(3e3)):
    obs = env.reset()
    memory = []
    for t in range(int(1e10)):
        #env.render()
        #print(obs)
        #qn.printEvery(1000,t)

        # get action
        seed = np.random.rand()
        if seed > epsilonUp/(epsilonUp+i_episode):
            action_q = [sess.run(q.value,feed_dict={q.observation:[list(obs)+[0,0,0,0]]})[0][0],
            sess.run(q.value,feed_dict={q.observation:[[0,0]+list(obs)+[0,0]]})[0][0],
            sess.run(q.value,feed_dict={q.observation:[[0,0]+[0,0]+list(obs)]})[0][0]]
            action = np.argmax(action_q)
        else:
            action = env.action_space.sample()
        newObs, reward, done, info = env.step(action)
        memory.append([obs,action,reward])
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    for i, unit in enumerate(memory):
        G = np.sum([item[2] for j,item in enumerate(memory[i:])])
        unit.append(G)

    # if i_episode < 1500:
    #     batch_size = 50
    # elif i_episode < 2000:
    #     batch_size = 20
    # elif i_episode < 3000:
    #     batch_size = 5
    # else:
    #     batch_size = 2

    for chunk in qn.chunks(memory,batch_size):
        obsChunk, actionChunk, _, targetChunk = list(zip(*chunk))
        tObs = np.zeros((len(obsChunk),len(obsChunk[0])*3))
        for i,obs in enumerate(obsChunk):
            action = actionChunk[i]
            tObs[i,action*2:action*2+2] = obs
        targetChunk = [[x] for x in targetChunk]
        sess.run(q.train_value,feed_dict={q.observation:tObs,
        q.target:targetChunk})
