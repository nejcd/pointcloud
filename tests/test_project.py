import unittest

import numpy as np

import pointcloud.utils.misc as misc
from pointcloud.pointcloud import PointCloud
from pointcloud.project import Project
from pointcloud.tile import Tile

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

        names_polygons = misc.get_names_and_polygons_in_workspace(workspace,
                                                                  settings={'step': 25, 'x_pos': 3, 'y_pos': 4})

        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        tile_names = pointcloud.get_tiles()
        tile = pointcloud.get_tile(list(tile_names.keys())[0])
        points = tile.get_points()
        labels = tile.get_labels()
        self.assertIsInstance(tile, Tile)
        self.assertEqual((11089, 3), np.shape(points))
        self.assertEqual((11089, 1), np.shape(labels))

    def test_get_project_bbox(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name)
        names_polygons = misc.get_names_and_polygons_in_workspace(workspace,
                                                                  settings={'step': 25, 'x_pos': 3, 'y_pos': 4})
        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        bbox = project.get_project_bbox()
        print(bbox)

    def test_get_stats(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name)

        names_polygons = misc.get_names_and_polygons_in_workspace(workspace,
                                                                  settings={'step': 25, 'x_pos': 3, 'y_pos': 4})
        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        stats = project.get_stats()
        true_values = {'name': 'Test',
                       'num_pointclouds': 1,
                       'workspace': 'test_data/',
                       'pointclouds': {'p1':
                                           {'area': 186.82999999999998,
                                            'num_points': 21931,
                                            'density': 117.38,
                                            'tiles': 2}}}
        self.assertEqual(true_values, stats)


if __name__ == '__main__':
    unittest.main()
