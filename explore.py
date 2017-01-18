import gym
import numpy as np
from numpy.matlib import repmat

env = gym.make('MountainCar-v0')

def discretize(span,n,overlap=1/10):
    space, step = np.linspace(span[0],span[1],n+1,retstep=True)
    overlapSize = step*overlap
    overlapUnit = overlapSize/2
    left = np.copy(space[:-1])
    right = np.copy(space[1:])
    left -= overlapUnit
    left[0] += overlapUnit
    right += overlapUnit
    right[-1] -= overlapUnit
    return np.stack([left,right]).T
    
def binarize(vals,discreteSpan):
    # vals is a 1-d numpy array not matrix
    n = discreteSpan.shape[0]
    valCount = vals.shape[0]
    valsRep = repmat(vals,n,1)
    left = valsRep >= repmat(discreteSpan[:,0,None],1,valCount)
    right = valsRep <= repmat(discreteSpan[:,1,None],1,valCount)
    return np.logical_and(left,right).astype(np.float32)

    
    

span = [-1.2, 0.6]