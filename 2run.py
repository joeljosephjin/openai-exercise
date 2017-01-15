import gym
from gym import wrappers
import shutil
import os
import numpy as np
import qn

from simple import models


for i_episode in range(500):
    obs = env.reset()
    for t in range(500):
        #env.render()
        #print(observation)

        # get action
        Qs = [model.predict(obs) for model in models]
        action = np.argmax(Qs)
        newObs, reward, done, info = env.step(action)
        obs = newObs
        if done:
            print("{}th episode finished after {} timesteps"\
            .format(i_episode, t+1))
            break
