import random
import sys
from pathlib import Path

import numpy as np
from pointcloud.tile import Tile
from pointcloud.utils import processing
from shapely.geometry import MultiPolygon


class PointCloud:
    def __init__(self, name, workspace, epsg=None, metadata=None, file_format='las', file_format_settings=None):
        """

        :param name:
        :param workspace: Path
        :param epsg:
        :param metadata:
        :param file_format:
        :param file_format_settings:
        """
        self.file_format_settings = file_format_settings
        self.metadata = metadata
        self.epsg = epsg
        self.workspace = Path(workspace)
        self.name = name
        self.stats = None
        self.tiles = {}
        self.polygons = None
        self.train_tiles = {}
        self.test_tiles = {}
        self.labels = {}
        self.file_format = file_format

    def get_name(self):
        return self.name

    def set_file_format_settings(self, file_format_settings):
        self.file_format_settings = file_format_settings

    def set_labels(self, labels):
        self.labels = labels

    def get_labels(self):
        return self.labels

    def add_tile(self, name, polygon=None):
        if name is self.tiles:
            raise ValueError('Tile with that name already exists')

        self.tiles[name] = Tile(name, polygon=polygon, workspace=self.workspace, file_format=self.file_format,
                                file_format_settings=self.file_format_settings)

    def create_new_tile(self, name, points, labels=None, features=None):
        """
        :param features:
        :param labels:
        :param name:
        :param points:
        :return:
        """
        polygon = processing.boundary(points)
        tile = Tile(name, polygon, self.workspace, file_format=self.file_format,
                    file_format_settings=self.file_format_settings)
        tile.set_points(points)

        if labels is not None:
            tile.set_labels(labels)
        if features is not None:
            tile.set_features(features)

        tile.store()
        self.add_tile(name, polygon)

    def number_of_tiles(self):
        return len(self.tiles)

    def get_tile(self, name):
        if self.tiles[name] is None:
            return None

        return self.tiles[name]

    def get_tiles(self):
        return self.tiles

    def get_train_tiles_names(self):
        return self.train_tiles

    def get_test_tiles_names(self):
        return self.test_tiles

    def get_train_tiles(self):
        return {tile: self.tiles[tile] for tile in self.get_train_tiles_names()}

    def get_test_tiles(self):
        return {tile: self.tiles[tile] for tile in self.get_test_tiles_names()}

    def create_train_test_split(self, train=0.8, seed=800815):
        """
        Create train test split trough tiles
        :param train:
        :param seed:
        :return:
        """
        print(PendingDeprecationWarning('Remove in future versions. (Use misc.create_train_test_split())'))

        train_num = int(np.ceil(len(self.tiles) * train))
        test_num = int(np.ceil(len(self.tiles) * (1 - train)))
        if train_num + test_num != len(self.tiles):
            diff = len(self.tiles) - (test_num + train_num)
            test_num = test_num + diff

        random.seed(seed)
        keys = list(self.tiles.keys())
        random.shuffle(keys)
        self.train_tiles = keys[0:train_num]
        self.test_tiles = keys[train_num:train_num + test_num]
        return self.train_tiles, self.test_tiles

    def get_tiles_by_point_count(self, number_of_points):
        """
        Get only tiles with more then x number of points
        :param number_of_points:
        :return:
        """
        tile_list = {}
        for tile_name in self.tiles:
            tile = self.tiles[tile_name]
            if tile.get_number_of_points() >= number_of_points:
                tile_list[tile_name] = tile
        return tile_list

    def remove_tiles_by_point_count(self, number_of_points):
        """
        Remove Tiles where point count lower then specified
        :param number_of_points:
        :return:
        """
        for tile_name in list(self.tiles.keys()):
            tile = self.tiles[tile_name]
            if tile.get_number_of_points() < number_of_points:
                del self.tiles[tile_name]

    def calculate_tile_polygons_from_points(self):
        """
        Recalculate polygons around points
        :return:
        """
        n = 1
        for name, tile in self.tiles.items():
            sys.stdout.write('\r Processing polygons: {0}/{1}'.format(n, len(self.tiles)))
            sys.stdout.flush()
            tile.calculate_tile_polygon_from_points()
            n += 1
        self.reset_stats()

    def get_stats(self):
        """
        Get stats for Point Cloud
        :return:
        """
        if self.stats is None:
            self.stats = self.calulcate_stats()

        return self.stats

    def reset_stats(self):
        """
        Reset all stats
        :return:
        """
        self.stats = None

    def calulcate_stats(self):
        """
        :return:
        """
        stats = {
            'area': 0,
            'num_points': 0,
            'density': 0,
            'tiles': len(self.tiles),
            'class_frequency': None
        }
        for name, tile in self.tiles.items():
            stats['area'] += round(tile.get_area(), 2)
            stats['num_points'] += tile.get_number_of_points()

        stats['density'] = round(stats['num_points'] / (stats['area'] + 1e-9), 2)
        stats['class_frequency'] = self.calculate_point_count_per_class()

        return stats

    def calculate_point_count_per_class(self):
        fq = {}
        for tile_name in list(self.tiles.keys()):
            tile = self.tiles[tile_name]
            fq_tile = tile.get_point_count_per_class()

            for c, count in fq_tile.items():
                if c in fq:
                    fq[int(c)] += count
                else:
                    fq[int(c)] = count
        return fq

    def get_intersected_tiles(self, geometry):
        """
        :param geometry:
        :return:
        """
        intersected = []
        for name, tile in self.tiles.items():
            if tile.intersects(geometry):
                intersected.append(tile)

        return intersected

    def get_intersected_points(self, geometry):
        """
        :param geometry:
        :return:
        """
        tiles = self.get_intersected_tiles(geometry)
        points = None
        labels = None
        features = None
        for tile in tiles:
            clipped_points, clipped_labels, clipped_features = tile.clip(geometry)
            if len(clipped_points) != 0:
                if points is None:
                    points = clipped_points
                else:
                    points = np.append(points, clipped_points, axis=0)

            if len(clipped_labels) != 0:
                if labels is None:
                    labels = clipped_labels
                else:
                    labels = np.append(labels, clipped_labels, axis=0)

            if len(clipped_features) != 0:
                if features is None:
                    features = clipped_features
                else:
                    features = np.append(features, clipped_features, axis=0)

        return points, labels, features

    def get_polygons(self):
        """
        Get polygons of all pointclouds and tiles
        :return:
        """
        geometries = []
        for n, tile in self.tiles.items():
            geometries.append(tile.get_polygon())
        return MultiPolygon(geometries)
