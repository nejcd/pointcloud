import os
import glob
from pointcloud.pointcloud import PointCloud
from pointcloud.utils import misc
from shapely.geometry import MultiPolygon
import pickle
from pathlib import Path


class Project:
    ext = '.lp'

    def __init__(self, project_name, epsg=None, workspace='./', pointcloud=None, labels=None):
        """

        :param labels:
        :param project_name:
        :param epsg: 
        :param workspace: 
        :param pointcloud: 
        """
        self.labels = labels
        self.epsg = epsg
        self.workspace = workspace
        self.pointcloud = pointcloud
        self.name = project_name
        self.pointcloud = PointCloud(project_name, workspace, epsg)
        self.stats = None
        self.altclouds = {}

    def get_name(self):
        """
        :return:
        """
        return self.name

    def get_point_cloud(self):
        """
        :rtype: PointCloud
        """
        return self.pointcloud

    def create_tile_and_add_to_point_cloud(self, tile_name, tile_polygon):
        """
        :param tile_name:
        :param tile_polygon:
        """
        if self.pointcloud is None:
            raise UserWarning('Pointcloud not initialised, can not add tile')
        self.pointcloud.add_tile(tile_name, tile_polygon)

    def add_tiles_in_workspace(self, extension='las', polygon_from_filename_settings=None):
        """
        :param extension:
        :param polygon_from_filename_settings:
        """
        files = glob.glob(self.workspace + "*." + extension)
        if len(files) == 0:
            raise UserWarning('No files in current workspace')
        for file in files:
            file_name = file.split('/')[-1]
            if polygon_from_filename_settings is not None:
                step, x_pos, y_pos = self.get_polygon_from_file_settings(polygon_from_filename_settings)
                polygon = misc.calculate_polygon_from_filename(file_name, step, x_pos, y_pos)
            else:
                polygon = misc.calculate_polygon_from_file(file_name)

            self.create_tile_and_add_to_point_cloud(file_name, polygon)

    @staticmethod
    def get_polygon_from_file_settings(settings):
        """
        :param settings:
        :return:
        """
        try:
            return settings['step'], settings['x_pos'], settings['y_pos']
        except ValueError:
            print('Not Valid Settings')

    def calculate_tile_polygons_from_points(self):
        self.pointcloud.calculate_tile_polygons_from_points()
        self.reset_stats()

    def get_stats(self):
        if self.stats is None:
            self.stats = self.pointcloud.get_stats()

        return self.stats

    def reset_stats(self):
        self.stats = None

    def get_intersected_points(self, geometry):
        return self.pointcloud.get_intersected_points(geometry)

    def add_new_altcloud(self, name_altcloud, subfolder, epsg):
        if name_altcloud is self.altclouds:
            raise ValueError('Alt PointCloud with that name already exists')

        path = self.workspace + subfolder
        if not os.path.exists(path):
            os.makedirs(path)

        self.altclouds[name_altcloud] = PointCloud(name_altcloud, path, epsg)
        return self.altclouds[name_altcloud]

    def get_all_altclouds(self):
        return self.altclouds

    def get_altcloud(self, name_altcloud):
        if name_altcloud in self.altclouds:
            return self.altclouds[name_altcloud]
        return None

    def get_polygons(self):
        multi = MultiPolygon([tile.get_polygon() for n, tile in self.pointcloud.get_tiles().items()])
        return multi

    def get_project_bbox(self):
        return self.get_polygons().bounds

    def can_load(self):
        my_file = Path(self.workspace + self.name + self.ext)
        return my_file.is_file()

    def load(self):
        print('Loading project {0}'.format(self.name))
        f = open(self.workspace + self.name + self.ext, 'rb')
        tmp_dict = pickle.load(f)
        f.close()

        self.__dict__.update(tmp_dict)

    def save(self):
        print('Saving project {0}'.format(self.name))
        f = open(self.workspace + self.name + self.ext, 'wb')
        pickle.dump(self.__dict__, f, 2)
        f.close()

    def create_train_test_split(self, train=0.8, seed=800815):
        self.pointcloud.create_train_test_split(train=train, seed=seed)
