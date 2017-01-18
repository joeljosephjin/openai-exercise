import numpy as np
import tensorflow as tf
import Q as q
import qn
import util as u


import gym
from gym import wrappers
import shutil
import os

saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver.restore(sess, 'models/moutainCar-pretrain-td')

env = gym.make('MountainCar-v0')
# logDir = 'tmp/MountainCar-v0-td'
# if os.path.isdir(logDir):
#     shutil.rmtree(logDir)
# env = wrappers.Monitor(env, logDir)

keep_size = 2000
#gamma = 0.99
epsilonUp = 50
epsilonDn = 100

batch_size = 50
td_n = 20

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

    tObs = np.zeros((len(memory),len(memory[0][0])*3))
    rewards = []
    for i, item in enumerate(memory):
        rewards.append(item[2])
        action = item[1]
        tObs[i,action*2:action*2+2] = item[0]
    q_val = sess.run(q.value,feed_dict={q.observation:tObs})
    targets = u.getTD(q_val,rewards,td_n)

    for chunk in qn.chunks(range(len(memory)),batch_size):
        obsChunk = tObs[chunk,:]
        targetChunk = np.array(targets)[chunk]

        sess.run(q.train_value,feed_dict={q.observation:obsChunk,
        q.target:targetChunk})
