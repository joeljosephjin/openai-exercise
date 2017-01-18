import unittest
import rl

class TestRL(unittest.TestCase):

    def test_Discretizer(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_combineBinaryFeatures(self):
        x = np.array([[0,1,0],[1,0,0],[0,0,1]])
        y = np.array([[0,1],[1,0],[1,0]])
        tensor = rl.combineBinaryFeatures([x,y],ravel=False)
        self.assertAlmostEqual(tensor[0,1,1],1)
        self.assertAlmostEqual(tensor[0,1,0],0)
        self.assertAlmostEqual(tensor[0,0,1],1)

if __name__ == '__main__':
    unittest.main()
