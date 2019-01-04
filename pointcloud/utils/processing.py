import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def sample_to_target_size(pointcloud, target_size, shuffle=True):
    """
    Return N (target_size) of data and shuffles it
    Beware original NP Array Gets Shuffled Also!!!!

    :param pointcloud:
    :param target_size:
    :param shuffle:
    :return:
    """
    if shuffle:
        np.random.shuffle(pointcloud)

    return pointcloud[:target_size, :]


def split_points_labels(pointcloud, points_from=0, points_to=3, labels=3):
    """

    :param pointcloud:
    :param points_from:
    :param points_to:
    :param labels:
    :return:
    """
    return pointcloud[:, points_from:points_to], pointcloud[:, labels]


def scale_points(pointcloud, scale):
    """
    Scales Points

    :param pointcloud:
    :param scale:
    :return:
    """
    return np.multiply(pointcloud, scale)


def translate_points(pointcloud, translation):
    """

    :param pointcloud:
    :param translation:
    :return:
    """
    return np.add(pointcloud, translation)


def translate_and_scale_from_bounds(pointcloud, bounds):
    """

    :param pointcloud:
    :param bounds:
    :return:
    """
    dx = bounds[1][0] - bounds[0][0]
    dy = bounds[1][1] - bounds[0][1]
    dz = bounds[1][2] - bounds[0][2]

    mean_x = 0.5 * (bounds[0][0] + bounds[1][0])
    mean_y = 0.5 * (bounds[0][1] + bounds[1][1])
    mean_z = 0.5 * (bounds[0][2] + bounds[1][2])
    max_value = max([dx, dy, dz])

    pointcloud = translate_points(pointcloud, [-mean_x, -mean_y, -mean_z])
    pointcloud = scale_points(pointcloud, [1/max_value, 1/max_value, 1/max_value])

    return pointcloud


def boundary(points, show=False):
    """
    :param points:
    :param show:
    :return:
    """
    hull = ConvexHull(points[:, 0:2])
    vertices = np.zeros([len(hull.vertices), 2])
    for i, vertex in enumerate(hull.vertices):
        vertices[i][0] = points[vertex][0]
        vertices[i][1] = points[vertex][1]

    if show:
        for simplex in hull.simplices:
            plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
        plt.show()

    return Polygon(vertices)


def clip_by_bbox(points, bbox):
    """
    :param points:
    :param bbox:
    :return:
    """
    points = points[bbox[0] < points[:, 0]]
    points = points[bbox[2] > points[:, 0]]
    points = points[bbox[1] < points[:, 1]]
    points = points[bbox[3] > points[:, 1]]
    return points



if __name__ == '__main__':
    test_pc = np.array([[1, 1, 1, 0],
                        [1, 2, 1, 0],
                        [3, 1, 1, 0],
                        [4, 5, 1, 0],
                        [3, 6, 10, 1],
                        [2, 5, 10, 1],
                        [4, 6, 10, 1],
                        [3, 5, 10, 1]])

    test_pc2 = np.array([[0, 0, 1, 0],
                        [0, 10, 1, 0],
                        [10, 0, 1, 0],
                        [10, 10, 1, 0],
                        [3, 6, 10, 1],
                        [2, 5, 10, 1],
                        [4, 6, 10, 1],
                        [3, 5, 10, 1]])
    clip = Polygon([(0, 0), (0, 1), (2, 2), (0, 2)])
    clipped = clip_by_bbox(test_pc, clip.bounds)
    print(clipped)



