import numpy as np
import random
from pointcloud.tile import Tile
from pointcloud.utils import processing
import sys


class PointCloud:
    def __init__(self, name, workspace, epsg, metadata=None):
        self.metadata = metadata
        self.epsg = epsg
        self.workspace = workspace
        self.name = name
        self.stats = None
        self.tiles = {}
        self.polygons = None
        self.train_tiles = {}
        self.test_tiles = {}

    def get_name(self):
        return self.name

    def add_tile(self, name, polygon=None):
        if name is self.tiles:
            raise ValueError('Tile with that name already exists')

        self.tiles[name] = Tile(name, polygon=polygon, workspace=self.workspace)

    def create_new_tile(self, name, points):
        polygon = processing.boundary(points)
        tile = Tile(name + '.las', polygon, self.workspace)
        tile.store_new_tile(points)

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
        train_num = int(np.ceil(len(self.tiles) * train))
        test_num = int(np.ceil(len(self.tiles) * (1 - train)))
        if train_num + test_num != len(self.tiles):
            diff = len(self.tiles) - (test_num + train_num)
            test_num = test_num + diff

        random.seed(seed)
        keys = list(self.tiles.keys())
        random.shuffle(keys)
        self.train_tiles = keys[0:train_num]
        self.test_tiles = keys[train_num:train_num+test_num]
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
        Recalucalte polygons around points
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
            'tiles': len(self.tiles)
        }
        for name, tile in self.tiles.items():
            stats['area'] += round(tile.get_area(), 2)
            stats['num_points'] += tile.get_number_of_points()
        stats['density'] = round(stats['num_points'] / stats['area'], 2)

        return stats

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
        for tile in tiles:
            clipped = tile.clip(geometry)
            if len(clipped) != 0:
                if points is None:
                    points = clipped
                else:
                    points = np.append(points, clipped, axis=0)

        return points
