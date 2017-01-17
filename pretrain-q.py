import pickle
import numpy as np
#
# with open('data/mountain-car-1.pkl','rb') as mf:
#     total = pickle.load(mf)
#
# keep_size = 2000
# gamma = 1
#
# train = []
# for memory in total:
#     if len(memory) > keep_size:
#         memory = memory[-keep_size:]
#
#     for i, unit in enumerate(memory):
#         G = np.sum([x[2] for x in memory[i:]])
#         unit.append(G)
#     train.append(memory)
#
# with open('data/train.pkl','wb') as mf:
#      pickle.dump(train,mf)

with open('data/train.pkl','rb') as mf:
    train = pickle.load(mf)


import tensorflow as tf
import Q as q
import qn


saver = tf.train.Saver()
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

mses = []
batch_size = 50
for memory in train[:10]:
    for chunk in qn.chunks(memory,batch_size):
        obsChunk, actionChunk, _, targetChunk = list(zip(*chunk))
        tObs = np.zeros((len(obsChunk),len(obsChunk[0])*3))
        for i,obs in enumerate(obsChunk):
            action = actionChunk[i]
            tObs[i,action*2:action*2+2] = obs
        targetChunk = [[x] for x in targetChunk]
        mses.append(sess.run(q.mse,feed_dict={q.observation:tObs,
                                           q.target:targetChunk}))
        sess.run(q.train_value,feed_dict={q.observation:tObs,
        q.target:targetChunk})

saver.save(sess,'models/moutainCar-pretrain-10')

import matplotlib.pyplot as plt
plt.plot(mses)


import gym
env = gym.make('MountainCar-v0')

for i_episode in range(500):
    obs = env.reset()
    for t in range(int(1e10)):
        #env.render()
        #print(obs)
        #qn.printEvery(1000,t)

        # get action
        seed = np.random.rand()
        if seed > 0.1:
            action_q = [sess.run(q.value,feed_dict={q.observation:[list(obs)+[0,0,0,0]]})[0][0],
            sess.run(q.value,feed_dict={q.observation:[[0,0]+list(obs)+[0,0]]})[0][0],
            sess.run(q.value,feed_dict={q.observation:[[0,0]+[0,0]+list(obs)]})[0][0]]
            action = np.argmax(action_q)
        else:
            action = env.action_space.sample()
        newObs, reward, done, info = env.step(action)
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break
