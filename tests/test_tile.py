import unittest
from shapely.geometry import Polygon
from pointcloud.tile import Tile


class TestTile(unittest.TestCase):
    def test_create(self):
        tile = Tile('tile_1.las')
        self.assertEqual('tile_1.las', tile.get_name())

    def test_workspace(self):
        tile_1 = Tile('test_data/test_tile_27620_158050.las', workspace='test_data/')
        tile_2 = Tile('tile_2.las')

        self.assertEqual('test_data/', tile_1.get_workspace())
        self.assertEqual('./', tile_2.get_workspace())

    def test_polygon(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        tile_1 = Tile('tile_1.las', polygon=polygon)

        self.assertEqual(polygon, tile_1.get_polygon())

    def test_polygon(self):
        polygon = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060)])
        tile_1 = Tile('test_data/test_tile_27620_158050.las', polygon=polygon)

        self.assertEqual(polygon, tile_1.get_polygon())

    def test_area(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        tile_1 = Tile('tile_1.las', polygon=polygon)

        self.assertEqual(1, tile_1.get_area())

    def test_area_2(self):
        polygon = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060)])
        tile_1 = Tile('test_data/test_tile_27620_158050.las', polygon=polygon)

        self.assertEqual(100, tile_1.get_area())

    def test_bbox(self):
        polygon = Polygon([(0, 0), (0, 1), (1, 1), (1, 0)])
        tile_1 = Tile('tile_1.las', polygon=polygon)

        self.assertEqual((0.0, 0.0, 1.0, 1.0), tile_1.get_bbox())

    def test_number_of_points(self):
        tile_1 = Tile('test_data/test_tile_27620_158050.las')

        self.assertEqual(10842, tile_1.get_number_of_points())

    def test_density(self):
        polygon = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060)])
        tile_1 = Tile('test_data/test_tile_27620_158050.las', polygon=polygon)

        self.assertEqual(108.42, tile_1.get_density())



if __name__ == '__main__':
    unittest.main()
