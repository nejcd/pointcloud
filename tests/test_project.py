import unittest
from shapely.geometry import Polygon
from pointcloud.project import Project
import pointcloud.utils.misc as misc

name = 'Test'
workspace = '../tests'
epsg = '3912'
metadata = 'Testing pointcloud'


class ProjectTests(unittest.TestCase):

    def test_create(self):
        project = Project(name, workspace=workspace, epsg=epsg)
        misc.calculate_polygon_from_filename('test_data/test_tile_27620_158050.las', 10, 3, 4)

        self.assertEqual(name, project.get_name())

    def test_read_polygon(self):
        polygon = misc.calculate_polygon_from_filename('test_data/test_tile_27620_158050.las', 10, 3, 4)
        polygon_a = Polygon([(27620, 158050), (27630, 158050), (27630, 158060), (27620, 158060), (27620, 158050)]);

        self.assertEqual(polygon_a, polygon)


if __name__ == '__main__':
    unittest.main()
