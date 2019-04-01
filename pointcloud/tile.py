import laspy
import numpy as np

from pointcloud.utils import misc
from pointcloud.utils import processing


class Tile:
    def __init__(self, filename, polygon=None, workspace='./'):
        """
        Tile information for LIDAR tile

        :param filename:
        :param polygon (Shapely):
        :param workspace:
        """
        self.workspace = workspace
        self.filename = filename
        self.polygon = polygon
        self.area = None
        self.density = None
        self.number_of_points = None

    def get_name(self):
        return self.filename

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
        file = laspy.file.File(self.workspace + self.filename, mode='r')
        min_values = file.header.min
        max_values = file.header.max
        file.close()
        return min_values, max_values

    def get_polygon(self):
        return self.polygon

    def get_point_count_per_class(self):
        points = self.get_points()
        labels = points[:, -1]
        classes = np.unique(labels)

        fq = {}
        for c in classes:
            fq[c] = (labels == c).sum()

        return fq

    def get_number_of_points(self):
        if self.number_of_points is None:
            file = laspy.file.File(self.workspace + self.filename, mode='r')
            self.number_of_points = len(file.points)
            file.close()

        return self.number_of_points

    def get_density(self):
        if self.density is None:
            self.density = self.get_number_of_points() / self.get_area()

        return self.density

    def get_points(self):
        return misc.get_points(self.workspace + self.filename)

    def calculate_tile_polygon_from_points(self):
        self.polygon = processing.boundary(self.get_points())
        self.area = round(self.polygon.area, 2)

    def intersects(self, geometry):
        return self.polygon.intersects(geometry)

    def clip(self, clip_polygon):
        return processing.clip_by_bbox(self.get_points(), clip_polygon.bounds)

    def store_new_tile(self, points):
        header = laspy.header.Header()

        file_out = laspy.file.File(self.workspace + self.filename, mode='w', header=header)
        file_out.X = np.ndarray.astype(points[:, 0] * 100,
                                       dtype=int)  # TODO DO NICER :) ALSO HANDLE WHOLE HEADER STUFF ETC!
        file_out.Y = np.ndarray.astype(points[:, 1] * 100, dtype=int)
        file_out.Z = np.ndarray.astype(points[:, 2] * 100, dtype=int)
        file_out.classification = np.ndarray.astype(points[:, 3], dtype=int)

        # TODO THIS SHOULD GO AWAY
        file_out.header.offset = [0, 0, 0]
        file_out.header.scale = [0.01, 0.01, 0.01]

        file_out.close()


if __name__ == '__main__':
    file = laspy.file.File('../tests/test_tile_27620_158050.las', mode='r')
    print(file.header.scale)
    print(file.header.offset)
