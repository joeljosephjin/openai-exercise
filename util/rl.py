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
