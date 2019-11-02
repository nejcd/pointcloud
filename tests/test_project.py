import unittest
from pathlib import PosixPath

import numpy as np
import pointcloud.utils.misc as misc
from pointcloud.pointcloud import PointCloud
from pointcloud.project import Project, save_project, load_project
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
        pointcloud = project.add_new_pointcloud(point_cloud_name, file_format='las',
                                                file_format_settings=None)

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
        pointcloud = project.add_new_pointcloud(point_cloud_name, file_format='las')
        names_polygons = misc.get_names_and_polygons_in_workspace(workspace,
                                                                  settings={'step': 25, 'x_pos': 3, 'y_pos': 4})
        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        bbox = project.get_project_bbox()

    def test_get_stats(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name, file_format='las')

        names_polygons = misc.get_names_and_polygons_in_workspace(workspace,
                                                                  settings={'step': 25, 'x_pos': 3, 'y_pos': 4})
        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        stats = project.get_stats()
        true_values = {'name': 'Test',
                       'num_pointclouds': 1,
                       'workspace': PosixPath('test_data'),
                       'pointclouds': {'p1':
                                           {'area': 186.82999999999998,
                                            'class_frequency': {2: 17171, 3: 4760},
                                            'num_points': 21931,
                                            'density': 117.38,
                                            'tiles': 2}}}
        self.assertEqual(true_values, stats)

    def test_save_load(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud(point_cloud_name, file_format='las',
                                                file_format_settings=None)

        names_polygons = misc.get_names_and_polygons_in_workspace(workspace,
                                                                  settings={'step': 25, 'x_pos': 3, 'y_pos': 4})

        for data in names_polygons:
            pointcloud.add_tile(data['name'], data['polygon'])

        save_project(project)

        project_load = load_project(workspace + project_name + '.prj')

        self.assertEqual(project.get_name(), project_load.get_name())

        c0 = project.get_pointcloud(point_cloud_name)
        c1 = project_load.get_pointcloud(point_cloud_name)
        for data in names_polygons:
            t0 = c0.get_tile(data['name'])
            t1 = c1.get_tile(data['name'])
            self.assertEqual(t1.get_number_of_points(), t0.get_number_of_points())

    def test_get_tile_from_cloud_tile_name(self):
        project = Project(project_name, workspace=workspace, epsg=epsg)
        pointcloud = project.add_new_pointcloud('test_cloud', file_format='las',
                                                file_format_settings=None)

        pointcloud.add_tile('test_tile_1')
        pointcloud.add_tile('test_tile_2')

        t1_1 = pointcloud.get_tile('test_tile_1')
        t2_1 = pointcloud.get_tile('test_tile_2')

        t1_2 = project.get_tile_from_cloud_tile_name('test_cloud/test_tile_1')
        t2_2 = project.get_tile_from_cloud_tile_name('test_cloud/test_tile_2')
        self.assertEqual(t1_1, t1_2)
        self.assertEqual(t2_1, t2_2)


if __name__ == '__main__':
    unittest.main()
