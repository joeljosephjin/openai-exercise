import unittest
import rl
import pa
import numpy as np
import gym


class TestRL(unittest.TestCase):
    def test_getMC(self):
        r = [0,0,0,1]
        targets = rl.getMC(r)
        self.assertAlmostEqual(np.sum(targets),4)
        targets = rl.getMC(r,0.5)
        self.assertAlmostEqual(targets[-1][0],1)
        self.assertAlmostEqual(targets[-2][0],0.5)
        self.assertAlmostEqual(targets[-3][0],0.25)

        r = [1,1,1,1]
        targets = rl.getMC(r)
        self.assertAlmostEqual(targets[-2][0],2)
        targets = rl.getMC(r,0.5)
        self.assertAlmostEqual(targets[-2][0],1.5)
        self.assertAlmostEqual(targets[0][0],1.875)


    def test_Discretizer(self):
        d = rl.Discretizer([1,5],4)
        self.assertAlmostEqual(d.segments[0][0],1)
        self.assertAlmostEqual(d.segments[-1][1],5)
        self.assertAlmostEqual(d.segments[0][1],2.05)
        self.assertAlmostEqual(d.segments[-2][0],2.95)
        self.assertAlmostEqual(len(d.segments),4)

        vals = np.array([1.5,2,3.2])
        bVals = d.encode(vals)
        self.assertEqual(vals.shape[0],bVals.shape[0])
        self.assertEqual(d.n,bVals.shape[1])
        self.assertEqual(np.sum(bVals[0]),1)
        self.assertEqual(np.sum(bVals[1]),2)
        self.assertEqual(np.sum(bVals[2]),1)
        self.assertEqual(bVals[0,0],True)
        self.assertEqual(bVals[2,2],True)
        self.assertEqual(bVals[1,0],True)
        self.assertEqual(bVals[1,1],True)

    def test_OneHotEncoder(self):
        h = rl.OneHotEncoder([0,1,2,3])

        vals = np.array([1,3,0,2])
        bVals = h.encode(vals)
        self.assertEqual(bVals.shape[0],len(vals))
        self.assertEqual(bVals.shape[1],bVals.shape[1])
        self.assertEqual(np.sum(bVals),4)
        self.assertEqual(bVals[0,1],1)
        self.assertEqual(bVals[1,3],1)
        self.assertEqual(bVals[2,0],1)
        self.assertEqual(bVals[3,2],1)

    def test_combineBinaryFeatures(self):
        x = np.array([[0,1,0],[1,0,0],[0,0,1]]).astype(np.bool)
        y = np.array([[0,1],[1,0],[1,0]]).astype(np.bool)
        tensor = rl.combineBinaryFeatures([x,y],ravel=False)
        self.assertAlmostEqual(tensor[0,1,1],1)
        self.assertAlmostEqual(tensor[0,1,0],0)
        self.assertAlmostEqual(tensor[1,0,0],1)

        tensor = rl.combineBinaryFeatures([x,y])
        self.assertAlmostEqual(tensor[2,-2],1)

        y = np.array([[0,1],[1,1],[1,0]]).astype(np.bool)
        z = np.array([[0,1],[1,0],[1,0]]).astype(np.bool)
        tensor = rl.combineBinaryFeatures([x,y,z],ravel=False)
        self.assertAlmostEqual(tensor[1,0,0,0],1)
        self.assertAlmostEqual(tensor[1,0,1,0],1)

        y = np.array([[0,1],[1,1],[1,0]]).astype(np.bool)
        z = np.array([[0,1],[1,1],[1,1]]).astype(np.bool)
        tensor = rl.combineBinaryFeatures([y,z],ravel=False)
        self.assertAlmostEqual(np.sum(tensor[0]),1)
        self.assertAlmostEqual(np.sum(tensor[1]),4)
        self.assertAlmostEqual(np.sum(tensor[2]),2)

class testPA(unittest.TestCase):
    def test_SpaceEncoder(self):
        env = gym.make('MountainCar-v0')
        encoder = pa.SpaceEncoder(env,[20,10])

        action = np.array([0,1,0,2])
        bAction = encoder.encodeAction(action)
        self.assertEqual(bAction[0,0],True)
        self.assertEqual(bAction[3,2],True)

        obs = np.array([[-0.5,0.03],[0,-0.028],[0.06,0.042]])
        bObs = encoder.encodeObs(obs,ravel=False)
        self.assertAlmostEqual(np.sum(bObs[0]),1)
        self.assertAlmostEqual(np.sum(bObs[1]),2)
        self.assertAlmostEqual(np.sum(bObs[2]),4)
        self.assertAlmostEqual(bObs[0,7,7],1)
        self.assertAlmostEqual(bObs[1,-7,2],1)
        self.assertAlmostEqual(bObs[1,-7,3],1)
        self.assertAlmostEqual(bObs[2,-7,-2],1)
        self.assertAlmostEqual(bObs[2,-6,-3],1)

        action = np.array([1,0,2])
        bFeature = encoder.encodeObsAction(obs,action,ravel=False)
        self.assertAlmostEqual(np.sum(bFeature[0]),1)
        self.assertAlmostEqual(np.sum(bFeature[1]),2)
        self.assertAlmostEqual(np.sum(bFeature[2]),4)
        self.assertAlmostEqual(bFeature[0,7,7,1],1)
        self.assertAlmostEqual(bFeature[0,7,7,1],1)
        self.assertAlmostEqual(bFeature[1,-7,2,0],1)
        self.assertAlmostEqual(bFeature[1,-7,3,0],1)
        self.assertAlmostEqual(bFeature[2,-7,-3,2],1)
        self.assertAlmostEqual(bFeature[2,-6,-2,2],1)



if __name__ == '__main__':
    unittest.main()
