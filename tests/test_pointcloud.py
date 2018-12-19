import unittest
from shapely.geometry import Polygon, Point
from pointcloud.pointcloud import PointCloud


name = 'Test'
workspace = '../tests/'
epsg = '3912'
metadata = 'Testing pointcloud'
polygon_1 = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060)])
polygon_2 = Polygon([(27620, 158060), (27630, 158060), (27630, 158070), (27620, 158070)])
tile_name_1 = 'test_data/test_tile_27620_158050.las'
tile_name_2 = 'test_data/test_tile_27620_158060.las'


class ProjectTests(unittest.TestCase):

    def test_create(self):
        point_cloud = PointCloud(name, workspace, epsg, metadata)
        self.assertEqual(name, point_cloud.get_name())

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

        num_of_points = 11000  #tile1 has less, tile2 has more

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



if __name__ == '__main__':
    unittest.main()
