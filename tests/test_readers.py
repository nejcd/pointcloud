import unittest

import numpy as np
from pointcloud.utils import readers

file_format_settings = {'points': [0, 1, 2], 'labels': [3], 'features': [4, 5, 6]}

class TestTile(unittest.TestCase):

    def test_read_las_points(self):
        reader = readers.LasReader()
        points = reader.get_points('test_data/test_tile_27620_158050')
        self.assertEqual((10842, 3), np.shape(points))

    def test_read_las_intensity_and_labels_points(self):
        reader = readers.LasReader(settings={'points': 'scaled',
                                                 'labels': True,
                                                 'features': ['intensity', 'num_return', 'return_num']})
        points = reader.get_points('test_data/test_tile_27620_158050')
        labels = reader.get_labels('test_data/test_tile_27620_158050')
        features = reader.get_features('test_data/test_tile_27620_158050')
        self.assertEqual((10842, 3), np.shape(points))
        self.assertEqual((10842, 1), np.shape(labels))
        self.assertEqual((10842, 3), np.shape(features))

    def test_read_txt_points(self):
        reader = readers.TxtReader(settings=file_format_settings)
        points = reader.get_points('test_data/test_tile_27620_158050')
        self.assertEqual((10842, 3), np.shape(points))

    def test_read_txt_labels(self):
        reader = readers.TxtReader(settings=file_format_settings)
        labels = reader.get_labels('test_data/test_tile_27620_158050')
        self.assertEqual(10842, np.shape(labels)[0])

    def test_read_txt_features(self):
        reader = readers.TxtReader(settings=file_format_settings)
        features = reader.get_features('test_data/test_tile_27620_158050')
        self.assertEqual((10842, 3), np.shape(features))

    def test_read_npy_points(self):
        reader = readers.NpyReader(settings=file_format_settings)
        points = reader.get_points('test_data/test_tile_27620_158050')
        self.assertEqual((10842, 3), np.shape(points))

    def test_read_npy_labels(self):
        reader = readers.NpyReader(settings=file_format_settings)
        labels = reader.get_labels('test_data/test_tile_27620_158050')
        self.assertEqual(10842, np.shape(labels)[0])

    def test_read_npy_features(self):
        reader = readers.NpyReader(settings=file_format_settings)
        features = reader.get_features('test_data/test_tile_27620_158050')
        self.assertEqual((10842, 3), np.shape(features))


if __name__ == '__main__':
    unittest.main()
