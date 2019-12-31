import numpy as np
from pathlib import Path

from pointcloud.utils import processing, readers


class Tile:
    def __init__(self, name, polygon=None, workspace='./', file_format='las', file_format_settings=None, area=None,
                 density=None, number_of_points=None):
        """
        Tile information for LIDAR tile

        :param name:
        :param polygon (Shapely):
        :param workspace:
        """
        self.name = name
        self.polygon = polygon
        self.workspace = Path(workspace)
        self.area = area
        self.density = density
        self.number_of_points = number_of_points
        self.file_format = file_format
        self.file_format_settings = file_format_settings

        if self.file_format == 'las' or self.file_format == 'laz':
            self.reader = readers.LasReader(path=self.get_path(), settings=file_format_settings, extension=self.file_format)
        elif self.file_format == 'txt':
            self.reader = readers.TxtReader(path=self.get_path(), settings=file_format_settings)
        elif self.file_format == 'npy' or self.file_format == 'npz':
            self.reader = readers.NpyReader(path=self.get_path(), settings=file_format_settings, extension=self.file_format)
        else:
            raise Exception('File format not supported')

    def meta_data(self):
        polygon_wkt = None
        if self.polygon:
            polygon_wkt = self.polygon.wkt
        return {
            'name': self.name,
            'polygon': polygon_wkt,
            'workspace': str(self.workspace),
            'area': self.area,
            'density': self.density,
            'number_of_points': self.number_of_points,
            'file_format': self.file_format,
            'file_format_settings': self.file_format_settings
        }

    def get_file_format(self):
        """

        :return:
        """
        return self.file_format

    def get_name(self):
        """

        :return:
        """
        return self.name

    def set_file_format_settings(self, file_format_settings):
        """

        :param file_format_settings:
        :return:
        """
        self.file_format_settings = file_format_settings
        if self.file_format == 'las' or self.file_format == 'laz':
            self.reader = readers.LasReader(path=self.get_path(), settings=file_format_settings, extension=self.file_format)
        elif self.file_format == 'txt':
            self.reader = readers.TxtReader(path=self.get_path(), settings=file_format_settings)
        elif self.file_format == 'npy' or self.file_format == 'npz':
            self.reader = readers.NpyReader(path=self.get_path(), settings=file_format_settings, extension=self.file_format)

    def get_filename(self):
        """

        :return:
        """
        return '{0}.{1}'.format(self.get_name(), self.file_format)

    def get_workspace(self):
        """

        :return:
        """
        return self.workspace

    def get_area(self):
        """

        :return:
        """
        if self.polygon is None:
            raise ValueError('Polygon not set')

        return self.polygon.area

    def get_bbox(self):
        """

        :return:
        """
        if self.polygon is None:
            raise ValueError('Polygon not set')

        return self.polygon.bounds

    def get_min_max_values(self):
        """

        :return:
        """
        points = self.get_points()
        return np.min(points), np.max(points)

    def get_polygon(self):
        """

        :return:
        """
        return self.polygon

    def get_number_of_points(self):
        """

        :return:
        """
        if self.number_of_points is None:
            points = self.get_points()
            self.number_of_points = len(points)
        return self.number_of_points

    def set_number_of_points(self, number):
        self.number_of_points = number

    def get_density(self):
        """

        :return:
        """
        if self.density is None:
            self.density = self.get_number_of_points() / self.get_area()

        return self.density

    def get_points(self):
        """

        :return:
        """
        return self.reader.get_points()

    def get_labels(self):
        """

        :return:
        """
        return self.reader.get_labels()

    def get_features(self):
        """

        :return:
        """
        return self.reader.get_features()

    def get_all(self):
        """

        :return:
        """
        return self.reader.get_all()

    def get_path(self):
        """

        :return:
        """
        return self.get_workspace() / self.get_filename()

    def calculate_tile_polygon_from_points(self):
        """

        :return:
        """
        self.polygon = processing.boundary(self.get_points())
        self.area = round(self.polygon.area, 2)

    def intersects(self, geometry):
        """

        :param geometry:
        :return:
        """
        return self.polygon.intersects(geometry)

    def clip(self, clip_polygon):
        """

        :param clip_polygon:
        :return:
        """
        return processing.clip_by_bbox(self.get_points(), clip_polygon.bounds, labels=self.get_labels(), features=self.get_features())

    def store(self, points, labels, features):
        """

        :param points:
        :param labels:
        :param features:
        :return:
        """
        self.reader.store(points=points, labels=labels, features=features)
