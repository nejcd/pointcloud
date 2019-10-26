import unittest
from pointcloud.utils import octree
import numpy as np


class TestOcTree(unittest.TestCase):

    def test_create_tree(self):
        otree = octree.OcTree(np.ones([210, 3]))
        otree.grow()

if __name__ == '__main__':
    unittest.main()
