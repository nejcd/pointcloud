import unittest
from shapely.geometry import Polygon
from pointcloud.project import Project
import pointcloud.utils.misc as misc
from pointcloud.pointcloud import PointCloud
from pointcloud.tile import Tile
import numpy as np

project_name = 'Test'
point_cloud_name = 'p1'
workspace = 'test_data/'
epsg = '3912'
metadata = 'Testing pointcloud'

class ProjectTests(unittest.TestCase):

    def test_create(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)

        self.assertIsInstance(project, Project)
        self.assertEqual(project_name, project.get_name())

    def test_add_new_pointcloud(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name)

        self.assertIsInstance(pointcloud, PointCloud)
        self.assertEqual(point_cloud_name, pointcloud.get_name())

    def test_add_new_pointcloud_and_read_tiles(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name)

        names_polygons = misc.get_names_and_polygons_in_workspace(workspace, settings={'step': 25, 'x_pos': 3, 'y_pos': 4})
        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        tile_names = pointcloud.get_tiles()
        tile = pointcloud.get_tile(list(tile_names.keys())[0])
        points = tile.get_points()
        self.assertIsInstance(tile, Tile)
        self.assertEqual((11089, 4), np.shape(points))

    def test_get_project_bbox(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name)
        names_polygons = misc.get_names_and_polygons_in_workspace(workspace, settings={'step': 25, 'x_pos': 3, 'y_pos': 4})
        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        bbox = project.get_project_bbox()
        print(bbox)



if __name__ == '__main__':
    unittest.main()
