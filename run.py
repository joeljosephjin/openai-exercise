import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn


from policyGradient import observation, target, action, \
action_prob, train_value, train_action

sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

env = gym.make('CartPole-v0')
logDir = 'tmp/cartpole-experiment-1'
if os.path.isdir(logDir):
    shutil.rmtree(logDir)
env = wrappers.Monitor(env, logDir)

gamma = 0.9
batch_size = 10
keep_size = 20
memory = []
for i_episode in range(200):
    obs = env.reset()
    for t in range(500):
        #env.render()
        #print(observation)
        action_p = sess.run(action_prob, feed_dict={observation:obs})
        action = 0 if action_p > np.random.rand() else 1
        obs, reward, done, info = env.step(action)
        memory.append([obs,action,reward])
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break

    for i, unit in memory:
        G = [gamma**j*item[2] for item in memory[i:]]
        unit.append(G)

    if len(memory) > keep_size:
        memory = memory[-keep_size:]

    for chunk in
