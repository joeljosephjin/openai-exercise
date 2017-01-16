import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn
import pickle
#from simple import models

with open('models/mountain-car-simple.pkl','rb') as mf:
    models = pickle.load(mf)

env = gym.make('MountainCar-v0')
logDir = 'tmp/MountainCar-experiment-simple'
if os.path.isdir(logDir):
    shutil.rmtree(logDir)
env = wrappers.Monitor(env, logDir)

for i_episode in range(500):
    obs = env.reset()
    for t in range(int(1e10)):
        env.render()
        #print(obs)
        #qn.printEvery(1000,t)

        # get action
        if obs[1] < 1e-5:
            newObs, reward, done, info = env.step(0)
        elif obs[1] > 1e-5:
            newObs, reward, done, info = env.step(2)
        else:
            newObs, reward, done, info = env.step(1)
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break
