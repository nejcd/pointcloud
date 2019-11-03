import json
import os
import pickle
from pathlib import Path
from shapely.geometry import MultiPolygon
import shapely.wkt

from pointcloud.pointcloud import PointCloud
from pointcloud.tile import Tile
from pointcloud.utils import misc
import gc


def save_project(project):
    """

    :type project: Project
    :param project:
    :return:
    """
    with open('{:}/{:}.prj'.format(project.get_workspace(), project.get_name()), 'w') as outfile:
        json.dump(project.meta_data(), outfile)


def load_project(project_file_path):
    """

    :param project_file_path:
    :return:
    """
    with open(project_file_path, 'r') as read:
        p = json.load(read)

    project = Project(project_name=p['name'], epsg=p['epsg'], workspace=p['workspace'])

    for c in p['pointclouds']:
        pointcloud = PointCloud(name=c['name'], workspace=c['workspace'], file_format=c['file_format'],
                                file_format_settings=c['file_format_settings'],
                                labels_descriptions=c.get('labels_descriptions', None))
        project.add_pointcloud(pointcloud)
        for t in c['tiles']:
            polygon = None
            if t['polygon'] is not None:
                polygon = shapely.wkt.loads(t['polygon'])

            tile = Tile(name=t['name'], polygon=polygon, workspace=t['workspace'],
                        file_format=t.get('file_format', c['file_format']),
                        file_format_settings=t.get('file_format_settings', c['file_format_settings']), area=t.get('area', None),
                        density=t.get('density', None), number_of_points=t.get('number_of_points', None))
            pointcloud.add_tile(tile)

    return project


class Project:
    ext = 'lp'

    def __init__(self, project_name, epsg=None, workspace='./'):
        """
        :param project_name:
        :param epsg: 
        :param workspace: 
        """
        self.name = project_name
        self.workspace = Path(workspace)
        self.pointclouds = {}
        self.epsg = epsg
        self.train_pointclouds = None
        self.test_pointclouds = None
        self.stats = None

    def get_workspace(self):
        return self.workspace

    def meta_data(self):
        """

        :return:
        """
        return {
            'name': self.name,
            'workspace': str(self.workspace),
            'epsg': self.epsg,
            'stats': self.stats,
            'pointclouds': [cloud.meta_data() for _, cloud in self.pointclouds.items()]
        }

    def add_new_pointcloud(self, name, workspace=None, folder=None, file_format=None,
                           file_format_settings=None, labels_descriptions=None):
        """
        Adds PointClouds to project
        :param workspace:
        :param labels_descriptions:
        :param file_format_settings:
        :param file_format:
        :param name:
        :param folder:
        :return:
        """
        if name is self.pointclouds:
            raise ValueError('PointCloud with that name already exists')

        if workspace is None:
            if folder is None:
                workspace = self.workspace
            else:
                workspace = self.workspace / folder
                if not os.path.exists(workspace):
                    os.makedirs(workspace)

        pointcloud = PointCloud(name, workspace, self.epsg, file_format=file_format,
                                file_format_settings=file_format_settings, labels_descriptions=labels_descriptions)

        self.add_pointcloud(pointcloud)

        return pointcloud

    def add_pointcloud(self, pointcloud):
        """
        :type pointcloud: PointCloud
        :param pointcloud:
        :return:
        """
        if pointcloud.get_name() is self.pointclouds:
            raise ValueError('Pointcloud with that name already exists')

        self.pointclouds[pointcloud.get_name()] = pointcloud

    def get_name(self):
        """
        :return:
        """
        return self.name

    def get_pointclouds(self):
        """
        :rtype: PointCloud
        """
        return self.pointclouds

    def get_pointcloud(self, point_cloud_name):
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
        cloud = self.get_pointcloud(cloud_name)
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
                      'pointclouds': [cloud.get_stats() for _, cloud in self.pointclouds.items()]}

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
        raise DeprecationWarning('Use load_project()')

    def get_project_file_name(self):
        """
        :return:
        """
        return self.workspace / '{0}.{1}'.format(self.name, self.ext)

    def plot_project(self):
        for name, pointcloud in self.pointclouds.items():
            geometries = []
            for n, tile in pointcloud.get_tiles().items():
                geometries.append(tile.get_polygon())
            misc.plot_polygons(multipolygons=MultiPolygon(geometries), title=name)
