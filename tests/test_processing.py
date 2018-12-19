import unittest
from shapely.geometry import Polygon
import numpy as np
import pointcloud.utils.processing as processing

points = np.array([[1, 1, 1, 0],
                       [1, 2, 1, 0],
                       [3, 1, 1, 0],
                       [4, 5, 1, 0],
                       [3, 6, 10, 1],
                       [2, 5, 10, 1],
                       [4, 6, 10, 1],
                       [3, 5, 10, 1]])


class ProjectTests(unittest.TestCase):

    def test_sample_by_target_value(self):
        sampled = processing.sample_to_target_size(points, 4, shuffle=False)
        target = np.array([[1, 1, 1, 0],
                           [1, 2, 1, 0],
                           [3, 1, 1, 0],
                           [4, 5, 1, 0]])

        self.assertTrue((sampled == target).all())

    def test_clip_by_bbox(self):
        clip = Polygon([(0, 0), (2, 0), (2, 3), (0, 3)])
        clipped = processing.clip_by_bbox(points, clip.bounds)

        self.assertEqual(2, np.shape(clipped)[0])


if __name__ == '__main__':
    unittest.main()
