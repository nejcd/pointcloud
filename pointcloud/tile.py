import numpy as np
from pathlib import Path

from pointcloud.utils import processing, readers


class Tile:
    def __init__(self, name, polygon=None, workspace='./', file_format='las', file_format_settings=None):
        """
        Tile information for LIDAR tile

        :param name:
        :param polygon (Shapely):
        :param workspace:
        """
        self.points = None
        self.labels = None
        self.features = None
        self.workspace = Path(workspace)
        self.name = name
        self.polygon = polygon
        self.area = None
        self.density = None
        self.number_of_points = None

        if file_format == 'las':
            self.reader = readers.LasReader(settings=file_format_settings)
        elif file_format == 'txt':
            self.reader = readers.TxtReader(settings=file_format_settings)
        elif file_format == 'npy':
            self.reader = readers.NpyReader(settings=file_format_settings)
        else:
            raise Exception('File format not supported')

    def get_name(self):
        return self.name

    def get_filename(self):
        return '{0}.{1}'.format(self.get_name(), self.reader.extension)

    def get_workspace(self):
        return self.workspace

    def get_area(self):
        if self.polygon is None:
            raise ValueError('Polygon not set')

        return self.polygon.area

    def get_bbox(self):
        if self.polygon is None:
            raise ValueError('Polygon not set')

        return self.polygon.bounds

    def get_min_max_values(self):
        points = self.get_points()
        return np.min(points), np.max(points)

    def get_polygon(self):
        return self.polygon

    def get_point_count_per_class(self):
        labels = self.get_labels()
        classes = np.unique(labels)

        fq = {}
        for c in classes:
            fq[c] = (labels == c).sum()

        return fq

    def get_number_of_points(self):
        if self.number_of_points is None:
            points = self.get_points()
            self.number_of_points = len(points)
        return self.number_of_points

    def get_density(self):
        if self.density is None:
            self.density = self.get_number_of_points() / self.get_area()

        return self.density

    def get_points(self):
        if self.points is None:
            self.points = self.reader.get_points(self.get_path())
        return self.points

    def get_labels(self):
        if self.labels is None:
            self.labels = self.reader.get_labels(self.get_path())
        return self.labels

    def get_features(self):
        if self.features is None:
            self.features = self.reader.get_features(self.get_path())
        return self.features

    def get_data(self):
        return self.get_points(), self.get_labels(), self.get_features()

    def get_path(self):
        return self.workspace / self.get_filename()

    def calculate_tile_polygon_from_points(self):
        self.polygon = processing.boundary(self.get_points())
        self.area = round(self.polygon.area, 2)

    def intersects(self, geometry):
        return self.polygon.intersects(geometry)

    def clip(self, clip_polygon):
        return processing.clip_by_bbox(self.get_points(), clip_polygon.bounds, labels=self.get_labels(), features=self.get_features())

    def set_points(self, points):
        self.points = points

    def set_labels(self, labels):
        self.labels = labels

    def set_features(self, features):
        self.features = features

    def store(self):
        self.reader.store(path=self.get_path(), points=self.points, labels=self.labels, features=self.features)
