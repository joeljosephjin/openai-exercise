import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn
import tensorflow as tf
import pickle


env = gym.make('MountainCar-v0')
# logDir = 'tmp/cartpole-experiment-simple'
# if os.path.isdir(logDir):
#     shutil.rmtree(logDir)
# env = wrappers.Monitor(env, logDir)

gamma = 1
keep_size = 5000
batch_size = 20

total = []
for i_episode in range(int(2500)):
    obs = env.reset()
    memory = []
    for t in range(int(1e11)):
        #env.render()
        #print(observation)
        #qn.printEvery(100,t)

        # get action
        # action_prob = sess.run(p.action_prob,
        # feed_dict={p.observation:[obs]})
        #action = 1-int(np.round(action_prob))
        action = env.action_space.sample()
        newObs, reward, done, info = env.step(action)
        memory.append([list(obs),action,reward])
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    total.append(memory)

with open('data/mountain-car-1.pkl','wb') as mf:
    pickle.dump(total,mf)
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
    #     actionChunk = [np.float32(np.array([0,1,2])==x)
    #      for x in actionChunk]
    #     sess.run(p.train_action,feed_dict={p.observation:obsChunk,
    #     p.action:actionChunk, p.target:targetChunk})
    #     # sess.run(p.mse,feed_dict={p.observation:obsChunk,
    #     # p.target:targetChunk})
    #     sess.run(p.train_value,feed_dict={p.observation:obsChunk,
    #     p.target:targetChunk})


env.close()
