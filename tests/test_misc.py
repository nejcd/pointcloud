import unittest
import pointcloud.utils.misc as misc
from shapely.geometry import Polygon

workspace = 'test_data/'


class MiscTests(unittest.TestCase):

    def test_calculate_tile_size_from_target_number_of_points(self):
        grid = misc.calculate_tile_size_from_target_number_of_points(1000, 10, tile_type='grid')
        circle = misc.calculate_tile_size_from_target_number_of_points(1000, 10, tile_type='circle')

        self.assertEqual(10, grid)
        self.assertEqual(6, circle)

    def test_calculate_polygon_from_filename(self):
        polygon = misc.calculate_polygon_from_filename('test_data/test_tile_27620_158050', 10, 3, 4)
        polygon_a = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060), (27620, 158050)])

        self.assertEqual(polygon_a, polygon)

    def test_get_names_and_polygons_in_workspace(self):
        names_polygons = misc.get_names_and_polygons_in_workspace(workspace, settings={'step': 25, 'x_pos': 3, 'y_pos': 4})

        data1 = names_polygons[0]
        self.assertEqual('test_tile_27620_158060', data1['name'])
        self.assertIsInstance(data1['polygon'], Polygon)

        data1 = names_polygons[1]
        self.assertEqual('test_tile_27620_158050', data1['name'])
        self.assertIsInstance(data1['polygon'], Polygon)

if __name__ == '__main__':
    unittest.main()
