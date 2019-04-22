import unittest

import numpy as np
from pointcloud.utils import readers


class TestTile(unittest.TestCase):

    def test_read_las_points(self):
        reader = readers.LasReader()
        points = reader.get_points('test_data/test_tile_27620_158050.las')
        self.assertEqual((10842, 3), np.shape(points))

    def test_read_txt_points(self):
        reader = readers.TxtReader(xyz=[0, 1, 2])
        points = reader.get_points('test_data/test_tile_27620_158050.txt')
        self.assertEqual((10842, 3), np.shape(points))

    def test_read_txt_labels(self):
        reader = readers.TxtReader(xyz=[0, 1, 2], label=[3])
        labels = reader.get_labels('test_data/test_tile_27620_158050.txt')
        self.assertEqual(10842, np.shape(labels))

    def test_read_txt_features(self):
        reader = readers.TxtReader(xyz=[0, 1, 2], features=[4, 5, 6])
        features = reader.get_features('test_data/test_tile_27620_158050.txt')
        self.assertEqual((10842, 3), np.shape(features))

    def test_read_npy_points(self):
        reader = readers.NpyReader(xyz=[0, 1, 2])
        points = reader.get_points('test_data/test_tile_27620_158050.npy')
        self.assertEqual((10842, 3), np.shape(points))

    def test_read_npy_labels(self):
        reader = readers.NpyReader(xyz=[0, 1, 2], label=[3])
        labels = reader.get_labels('test_data/test_tile_27620_158050.npy')
        self.assertEqual(10842, np.shape(labels))

    def test_read_npy_features(self):
        reader = readers.NpyReader(xyz=[0, 1, 2], features=[4, 5, 6])
        features = reader.get_features('test_data/test_tile_27620_158050.npy')
        self.assertEqual((10842, 3), np.shape(features))


if __name__ == '__main__':
    unittest.main()
