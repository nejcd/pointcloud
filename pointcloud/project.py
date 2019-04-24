import os
import pickle
import random
from pathlib import Path

import numpy as np
from shapely.geometry import MultiPolygon

from pointcloud.pointcloud import PointCloud
from pointcloud.utils import misc


class Project:
    ext = 'lp'

    def __init__(self, project_name, epsg=None, workspace='./'):
        """
        :param project_name:
        :param epsg: 
        :param workspace: 
        :param pointcloud: 
        """
        self.name = project_name
        self.workspace = Path(workspace)
        self.pointclouds = {}
        self.epsg = epsg
        self.train_pointclouds = None
        self.test_pointclouds = None
        self.stats = None

    def add_new_pointcloud(self, name, folder=None, file_format=None, file_format_settings=None):
        """
        Adds PointClouds to project
        :param file_format_settings:
        :param file_format:
        :param name:
        :param folder:
        :return:
        """
        if name is self.pointclouds:
            raise ValueError('PointCloud with that name already exists')

        if folder is None:
            path = self.workspace
        else:
            path = self.workspace / folder
            if not os.path.exists(path):
                os.makedirs(path)

        self.pointclouds[name] = PointCloud(name, path, self.epsg, file_format=file_format,
                                            file_format_settings=file_format_settings)
        return self.pointclouds[name]

    def get_name(self):
        """
        :return:
        """
        return self.name

    def get_point_clouds(self):
        """
        :rtype: PointCloud
        """
        return self.pointclouds

    def get_point_cloud(self, point_cloud_name):
        """
        :rtype: PointCloud
        """
        if self.pointclouds[point_cloud_name] is None:
            raise UserWarning('Point clouds {0} not set'.format(point_cloud_name))

        return self.pointclouds[point_cloud_name]

    def get_tile_from_cloud_tile_name(self, cloud_tile_name, delimiter='/'):
        """
        Get tile from string (cloud/tile1)
        :param cloud_tile_name:
        :param delimiter:
        :return:
        """

        split = cloud_tile_name.split(delimiter)
        cloud_name = split[0]
        tile_name = split[1]
        cloud = self.get_point_cloud(cloud_name)
        tile = cloud.get_tile(tile_name)
        return tile


    def get_stats(self):
        """
        :return:
        """
        if self.stats is not None:
            return self.stats

        self.stats = {'name': self.name,
                      'num_pointclouds': len(self.pointclouds),
                      'workspace': self.workspace,
                      'pointclouds': {}}

        for name, pointcloud in self.pointclouds.items():
            self.stats['pointclouds'][name] = pointcloud.get_stats()

        return self.stats

    def reset_stats(self):
        """
        :return:
        """
        self.stats = None

    def get_polygons(self):
        """
        Get polygons of all pointclouds and tiles
        :return:
        """
        geometries = []
        for name, pointcloud in self.pointclouds.items():
            for n, tile in pointcloud.get_tiles().items():
                geometries.append(tile.get_polygon())
        return MultiPolygon(geometries)

    def get_project_bbox(self):
        """
        Get BBOX around project area
        :return:
        """
        return self.get_polygons().bounds

    def can_load(self):
        """
        Is there saved project version
        :return:
        """
        my_file = self.get_project_file_name()
        return my_file.is_file()

    def load(self):
        """
        Load saved project
        :return:
        """
        print('\nLoading project {0}'.format(self.name))
        f = open(self.get_project_file_name(), 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def get_project_file_name(self):
        """
        :return:
        """
        return self.workspace / '{0}.{1}'.format(self.name, self.ext)

    def save(self):
        """
        Save project
        :return:
        """
        print('\nSaving project {0}'.format(self.name))
        f = open((self.workspace / self.name).with_suffix('.' + self.ext), 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def plot_project(self):
        for name, pointcloud in self.pointclouds.items():
            geometries = []
            for n, tile in pointcloud.get_tiles().items():
                geometries.append(tile.get_polygon())
            misc.plot_polygons(multipolygons=MultiPolygon(geometries), title=name)
