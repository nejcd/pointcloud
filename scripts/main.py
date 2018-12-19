import numpy as np
import laspy
import sys

sys.path.insert(0, './pointcloud')
sys.path.insert(0, './pointcloud/utils')
from project import Project
import processing
import misc
import plot
import pickle

from shapely.geometry import Polygon

labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# workspace = '../../../datasets/flycom/kellag/subset_250/'
project_name = 'kelag_subset_250'
workspace = '/Users/nejc/deeppoint/datasets/flycom/kellag/tiles_25/'
project_name = 'tiles_25'

def create_project():
    return Project(project_name, workspace=workspace, epsg=9999, labels=labels)


def get_stats(project):
    stats = project.get_stats()
    print('Stats:', stats)

    tile_size = misc.calculate_tile_size_from_target_number_of_points(1014, stats['density'])
    print('TileSize', tile_size)
    bbox = project.get_project_bbox()
    project_polygons = project.get_polygons()
    print('BBOX', bbox)

    # plot.multipolygons(fishnet)



if __name__ == '__main__':
    project = create_project()
    if project.can_load():
        project.load()
    else:
        print('Creating project from tiles in workspace')
        project.add_tiles_in_workspace(polygon_from_filename_settings={'step': 250, 'x_pos': 3, 'y_pos': 4})
        project.calculate_tile_polygons_from_points()

    project.create_train_test_split(train=0.8)
    cloud = project.get_point_cloud()
    cloud.create_train_test_split()
    print(cloud.get_train_tiles_names())
    print(cloud.get_train_tiles())

    # project.save()
    #
    # # Create New Point cloud
    # cloud = project.get_point_cloud()
    # alt_cloud = project.add_new_altcloud('test_alt', 'tests/', epsg=9999)
    #
    # stats = project.get_stats()
    # tile_size = misc.calculate_tile_size_from_target_number_of_points(1014, stats['density'])
    # project_polygons = project.get_polygons()
    # fishnet = misc.create_fish_net(project_polygons.bounds, tile_size)
    #
    # for i, polygon in enumerate(fishnet):
    #     points = cloud.get_intersected_points(polygon)
    #
    #     if np.size(points) == 0:
    #         continue
    #
    #     alt_cloud.create_new_tile('points_{0}'.format(i), points)

