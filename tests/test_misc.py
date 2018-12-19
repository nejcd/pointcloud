import unittest
import pointcloud.utils.misc as misc


class MiscTests(unittest.TestCase):

    def test_calculate_tile_size_from_target_number_of_points(self):
        grid = misc.calculate_tile_size_from_target_number_of_points(1000, 10, tile_type='grid')
        circle = misc.calculate_tile_size_from_target_number_of_points(1000, 10, tile_type='circle')

        self.assertEqual(10, grid)
        self.assertEqual(6, circle)


if __name__ == '__main__':
    unittest.main()
