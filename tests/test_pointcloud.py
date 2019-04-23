import unittest

import numpy as np
from pointcloud.pointcloud import PointCloud
from shapely.geometry import Polygon, Point

name = 'Test'
workspace = '../tests/'
epsg = '3912'
metadata = 'Testing pointcloud'
polygon_1 = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060)])
polygon_2 = Polygon([(27620, 158060), (27630, 158060), (27630, 158070), (27620, 158070)])
tile_name_1 = 'test_data/test_tile_27620_158050'
tile_name_2 = 'test_data/test_tile_27620_158060'
points = np.array([[1., 1., 1.],
                   [1., 2., 1.],
                   [3., 1., 1.],
                   [4., 5., 1.],
                   [3., 6., 10.],
                   [2., 5., 10.],
                   [4., 6., 10.],
                   [3., 5., 10.]])
labels = np.array([0., 0., 0., 0., 1., 1., 1., 1.])
features = np.array([[1., 2., 1.],
                     [1., 2., 1.],
                     [1., 2., 1.],
                     [1., 2., 1.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.],
                     [1., 2., 3.]])

file_format_settings = {'points': [0, 1, 2], 'labels': [6], 'features': [3, 4, 5]}


class ProjectTests(unittest.TestCase):

    def test_create(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        self.assertEqual(name, point_cloud.get_name())

    def test_create_new_tiles(self):
        point_cloud = PointCloud(name, workspace='test_data/', metadata=metadata, file_format='txt',
                                 file_format_settings=file_format_settings)
        point_cloud.create_new_tile('test', points, labels=labels, features=features)
        tile = point_cloud.get_tile('test')

        points_1, labels_1, features_1 = tile.get_data()

        self.assertTrue((points == points_1).all())
        self.assertTrue((labels == labels_1).all())
        self.assertTrue((features == features_1).all())

    def test_add_tiles(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        point_cloud.add_tile(tile_name_1, polygon=polygon_1)
        point_cloud.add_tile(tile_name_1, polygon=polygon_1)

        self.assertEqual(1, point_cloud.number_of_tiles())

        point_cloud.add_tile(tile_name_2, polygon=polygon_2)
        self.assertEqual(2, point_cloud.number_of_tiles())

    def test_create_tiles(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        point_cloud.add_tile(tile_name_1, polygon_1)
        point_cloud.add_tile(tile_name_1, polygon_1)

        self.assertEqual(1, point_cloud.number_of_tiles())

        point_cloud.add_tile(tile_name_2, polygon_2)
        self.assertEqual(2, point_cloud.number_of_tiles())

    def test_get_tile(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        point_cloud.add_tile(tile_name_1, polygon_1)
        tile = point_cloud.get_tile(tile_name_1)

        self.assertEqual(tile_name_1, tile.get_name())

    def test_get_tiles_by_point_count(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        point_cloud.add_tile(tile_name_1, polygon_1)
        point_cloud.add_tile(tile_name_2, polygon_2)

        num_of_points = 11000  # tile1 has less, tile2 has more

        tiles = point_cloud.get_tiles_by_point_count(num_of_points)
        tile_name = next(iter(tiles))
        self.assertEqual(tile_name_2, tile_name)

    def test_get_intersected_tiles(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        point_cloud.add_tile(tile_name_1, polygon_1)
        point_cloud.add_tile(tile_name_2, polygon_2)

        self.assertEqual(0, len(point_cloud.get_intersected_tiles(Point(1, 1))))
        self.assertEqual(1, len(point_cloud.get_intersected_tiles(Point(27625, 158055))))
        self.assertEqual(2, len(point_cloud.get_intersected_tiles(Polygon([(27620, 158055),
                                                                           (27630, 158055),
                                                                           (27630, 158070),
                                                                           (27620, 158070)]))))

    def test_get_stats(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        point_cloud.add_tile(tile_name_1, polygon_1)

        true_res = {'area': 100.0, 'num_points': 10842, 'density': 108.42, 'tiles': 1,
                    'class_frequency': {2: 7386, 3: 3456}}
        self.assertEqual(true_res, point_cloud.get_stats())


if __name__ == '__main__':
    unittest.main()
