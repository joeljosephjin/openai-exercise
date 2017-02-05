import numpy as np
from numpy.matlib import repmat
import numpy as np


def getTD(q,r,n,gamma=1):
    count = len(q)
    targets = []
    for i in range(count):
        if i+n < count:
            qi = q[i+n]
            target = np.sum([gamma**j*item for j,item in enumerate(r[i:i+n])]
            + [gamma**(i+n)*qi])
        else:
            target = np.sum([gamma**j*item for j,item in enumerate(r[i:])])
        targets.append([target])
    return targets

def getMC(r,gamma=1):
    targets = []
    for i, _ in enumerate(r):
        target = np.sum([gamma**j*item for j,item in enumerate(r[i:])])
        targets.append([target])
    return targets

class Discretizer():
    # discretize conintuous space into binary feature space
    def __init__(self,span,n,overlap=1/10):
        # span: the range of the 1-D continous space
        # n: the number of segments span will be split into
        # overlap: how much two adjacent segments overlap with each other
        self.n = n
        self.span = span
        self.overlap = overlap
        space, step = np.linspace(span[0],span[1],n+1,retstep=True)
        overlapSize = step*overlap
        overlapUnit = overlapSize/2
        left = np.copy(space[:-1])
        right = np.copy(space[1:])
        left -= overlapUnit
        left[0] += overlapUnit
        right += overlapUnit
        right[-1] -= overlapUnit
        self.segments = np.stack([left,right]).T

    def encode(self,vals):
        # vals is a numpy array not matrix
        # return boolean
        # todo: check if vals is a matrix or array and transform accordingly
        # todo: check all the values in vals are within the range of self.span
        valCount = vals.shape[0]
        valsRep = repmat(vals,self.n,1)
        left = valsRep >= repmat(self.segments[:,0,None],1,valCount)
        right = valsRep <= repmat(self.segments[:,1,None],1,valCount)
        return np.logical_and(left,right).T

class OneHotEncoder():
    def __init__(self,span):
        self.span = np.array(span).astype(np.int)
        self.count = self.span.shape[0]

    def encode(self,vals):
        # return boolean
        vals = np.array(vals).astype(np.int)
        base = repmat(self.span,vals.shape[0],1)
        return (base.T == vals).T



def combineBinaryFeatures(features,ravel=True):
    # combine different binary features of the same set of observations
    # into a tensor
    # features: an array of binary features with first dimension equal
    # ravel: if ravel from 2nd dimension onwards into a vector
    obsCount = features[0].shape[0]
    dimCount = [feature.shape[1] for feature in features]
    tensor = np.zeros((obsCount,np.prod(dimCount)))
    for i in range(obsCount):
        left = features[0][i]
        for j in range(1,len(features)):
            left = np.dot(left[:,None],features[j][i][None,:])
            left = np.ravel(left)
        tensor[i,:] = left

    if not ravel:
        tensor = np.reshape(tensor,[obsCount]+dimCount)

    return tensor

# def combineBinaryFeatures(features,ravel=True):
#     # this implementation did not work because:
#     # np.array([[1,2],[3,4]])[np.array(True,True),np.array([True,True])]
#     # returns np.array([1,4]) instead of np.array([[1,2],[3,4]]) WIRED!
#     # combine different binary features of the same set of observations
#     # into a tensor
#     # features: an array of binary features with first dimension equal
#     # ravel: if ravel from 2nd dimension onwards into a vector
#     obsCount = features[0].shape[0]
#     dimCount = [feature.shape[1] for feature in features]
#     tensor = np.zeros([obsCount]+dimCount)
#     for i in range(obsCount):
#         string = 'tensor[i'
#         for j, _ in enumerate(features):
#             string += ',features[{}][i]'.format(j)
#         string += ']=1'
#         exec(string)
#
#     if ravel:
#         r_tensor = np.zeros((obsCount,np.prod(dimCount)))
#         for i in range(obsCount):
#             r_tensor[i] = np.ravel(tensor[i])
#         return r_tensor
#     else:
#         return tensor
