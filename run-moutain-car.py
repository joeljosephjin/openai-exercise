import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn
import tensorflow as tf

import policyGradient as p

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

env = gym.make('MountainCar-v0')
# logDir = 'tmp/cartpole-experiment-simple'
# if os.path.isdir(logDir):
#     shutil.rmtree(logDir)
# env = wrappers.Monitor(env, logDir)

gamma = 0.98
keep_size = 500
batch_size = 30

for i_episode in range(int(1e5)):
    obs = env.reset()
    memory = []
    # actions = []
    # action_probs = []
    for t in range(int(1e10)):
        #env.render()
        #print(observation)
        qn.printEvery(10000,t)

        # get action
        if i_episode < 50:
            action = env.action_space.sample()
        else:
            action_prob = sess.run(p.action_prob,
            feed_dict={p.observation:[obs]})
            #print(action_prob)
            #action = 1-int(np.round(action_prob))
            if np.random.rand() < action_prob[0][0]:
                action = 0
            elif np.random.rand() < action_prob[0][0] + action_prob[0][1]:
                action = 1
            else:
                action = 2
        # actions.append(action)
        # action_probs.append(action_prob[0])
        newObs, reward, done, info = env.step(action)
        memory.append([obs,action,1+reward])
        obs = newObs
        if done:
            memory[-1][2] = 1
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    for i, unit in enumerate(memory):
        G = np.sum([gamma**j*item[2] for j,item in enumerate(memory[i:])])
        unit.append(G)


    for chunk in qn.chunks(memory,batch_size):
        obsChunk, actionChunk, _, targetChunk = list(zip(*chunk))
        obsChunk = np.stack(obsChunk)
        targetChunk = [[x] for x in targetChunk]
        actionChunk = [np.float32(np.array([0,1,2])==x)
         for x in actionChunk]
        sess.run(p.train_action,feed_dict={p.observation:obsChunk,
        p.action:actionChunk, p.target:targetChunk})
        # sess.run(p.mse,feed_dict={p.observation:obsChunk,
        # p.target:targetChunk})
        sess.run(p.train_value,feed_dict={p.observation:obsChunk,
        p.target:targetChunk})
        print('learned')


env.close()
