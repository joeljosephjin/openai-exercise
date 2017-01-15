import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn
import tensorflow as tf

env = gym.make('')
# logDir = 'tmp/cartpole-experiment-simple'
# if os.path.isdir(logDir):
#     shutil.rmtree(logDir)
# env = wrappers.Monitor(env, logDir)

gamma = 0.5
keep_size = 20
batch_size = 10

for i_episode in range(500):
    obs = env.reset()
    memory = []
    for t in range(500):
        #env.render()
        #print(observation)

        # get action

        newObs, reward, done, info = env.step(action)
        memory.append([obs,action,reward])
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    # for i, unit in enumerate(memory):
    #     G = np.sum([gamma**j*item[2] for j,item in enumerate(memory[i:])])
    #     unit.append(G)
    #
    # if len(memory) > keep_size:
    #     memory = memory[-keep_size:]
    #
    # for chunk in qn.chunks(memory,batch_size):
    #     obsChunk, actionChunk, _, targetChunk = list(zip(*chunk))
    #     obsChunk = np.stack(obsChunk)
    #     targetChunk = [[x] for x in targetChunk]
    #     actionChunk = [[1,0] if x==0 else [0,1] for x in actionChunk]
    #     sess.run(p.train_action,feed_dict={p.observation:obsChunk,
    #     p.action:actionChunk, p.target:targetChunk})
    #     # sess.run(p.mse,feed_dict={p.observation:obsChunk,
    #     # p.target:targetChunk})
    #     sess.run(p.train_value,feed_dict={p.observation:obsChunk,
    #     p.target:targetChunk})

env.close()
