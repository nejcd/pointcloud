import unittest

import numpy as np
from shapely.geometry import Polygon

import pointcloud.utils.processing as processing

points = np.array([[1, 1, 1],
                   [1, 2, 1],
                   [3, 1, 1],
                   [4, 5, 1],
                   [3, 6, 10],
                   [2, 5, 10],
                   [4, 6, 10],
                   [3, 5, 10]])

labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

features = np.array([[1, 2, 1],
                     [1, 2, 1],
                     [1, 2, 1],
                     [1, 2, 1],
                     [1, 2, 3],
                     [1, 2, 3],
                     [1, 2, 3],
                     [1, 2, 3]])


class ProjectTests(unittest.TestCase):

    def test_sample_by_target_value(self):
        sampled_points, sampled_labels, sampled_features = processing.sample_to_target_size(points, 4, shuffle=False,
                                                                                            labels=labels,
                                                                                            features=features)

        target_points = np.array([[1, 1, 1],
                                  [1, 2, 1],
                                  [3, 1, 1],
                                  [4, 5, 1]])

        target_labels = np.array([0, 0, 0, 0])

        target_features = np.array([[1, 2, 1],
                                    [1, 2, 1],
                                    [1, 2, 1],
                                    [1, 2, 1]])

        self.assertTrue((sampled_points == target_points).all())
        self.assertTrue((sampled_labels == target_labels).all())
        self.assertTrue((sampled_features == target_features).all())

    def test_sample_by_target_value_random_shuffle(self):
        sampled_points, sampled_labels, sampled_features = processing.sample_to_target_size(points, 4, shuffle=True,
                                                                                            seed=0,
                                                                                            labels=labels,
                                                                                            features=features)

        target_points = np.array([[4, 6, 10],
                                  [3, 1, 1],
                                  [1, 2, 1],
                                  [3, 5, 10]])

        target_labels = np.array([1, 0, 0, 1])

        target_features = np.array([[1, 2, 3],
                                    [1, 2, 1],
                                    [1, 2, 1],
                                    [1, 2, 3]])

        self.assertTrue((sampled_points == target_points).all())
        self.assertTrue((sampled_labels == target_labels).all())
        self.assertTrue((sampled_features == target_features).all())

    def test_clip_by_bbox(self):
        clip = Polygon([(0, 0), (2, 0), (2, 3), (0, 3)])
        c_points, c_labels, c_features = processing.clip_by_bbox(points, clip.bounds, labels=labels, features=features)

        self.assertEqual((2, 3), np.shape(c_points))
        self.assertEqual((2,), np.shape(c_labels))
        self.assertEqual((2, 3), np.shape(c_features))

    def test_classify_close_by(self):
        new_labels = processing.classify_close_by(points, labels, from_label=0, to_label=1, close_to_label=1, radius=10)
        new_labels_same = processing.classify_close_by(points, labels, from_label=0, to_label=1, close_to_label=1, radius=9)
        target_new_labels = np.array([1, 1, 1, 1, 1, 1, 1, 1])

        self.assertEqual(new_labels, target_new_labels)
        self.assertEqual(new_labels_same, labels)


if __name__ == '__main__':
    unittest.main()
