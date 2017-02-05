# pa is abbreviation for openai
import rl
import numpy as np


class SpaceEncoder():
    def __init__(self,env,segCount):
        # currently only support observation of box class
        # and action of discrete class
        # segCount: an array of segment count for each dimension of observation
        spans = np.stack([env.observation_space.low,
        env.observation_space.high]).T
        self.obsEncoders = []
        for i,span in enumerate(spans):
            self.obsEncoders.append(
            rl.Discretizer(list(span),segCount[i]))
        self.actionEncoder = rl.OneHotEncoder(list(range(
        env.action_space.n)))

    def _getObsBinaries(self,obs):
        binaries = []
        for i, encoder in enumerate(self.obsEncoders):
            binaries.append(encoder.encode(obs[:,i]))
        return binaries

    def encodeObs(self,obs,ravel=True):
        binaries = self._getObsBinaries(obs)
        #print(binaries)
        return rl.combineBinaryFeatures(binaries,ravel)

    def encodeAction(self,action):
        return self.actionEncoder.encode(action)

    def encodeObsAction(self,obs,action, ravel=True):
        binaries = self._getObsBinaries(obs)
        actionBinary = self.encodeAction(action)
        return rl.combineBinaryFeatures(binaries+[actionBinary],ravel)
