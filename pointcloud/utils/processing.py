import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def sample_to_target_size(points, target_size, shuffle=True, labels=None, features=None, seed=None):
    """
    Return N (target_size) of data and shuffles it
    Beware original NP Array Gets Shuffled Also!!!!

    :param points:
    :param target_size:
    :param shuffle:
    :return:
    """
    if shuffle:
        np.random.seed(seed)
        s = np.arange(np.shape(points)[0])
        np.random.shuffle(s)
        points = np.array(points)[s]

        if labels is not None:
            labels = np.array(labels)[s]

        if features is not None:
            features = np.array(features)[s]

    return points[:target_size, :], labels[:target_size], features[:target_size, :]


def scale_points(points, scale):
    """
    Scales Points

    :param points:
    :param scale:
    :return:
    """
    return np.multiply(points, scale)


def translate_points(points, translation):
    """

    :param points:
    :param translation:
    :return:
    """
    return np.add(points, translation)


def translate_and_scale_from_bounds(points, bounds):
    """

    :param points:
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

    points = translate_points(points, [-mean_x, -mean_y, -mean_z])
    points = scale_points(points, [1/max_value, 1/max_value, 1/max_value])

    return points


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


def clip_by_bbox(points, bbox, labels=None, features=None):
    """
    :param features:
    :param labels:
    :param points:
    :param bbox:
    :return:
    """
    b0 = bbox[0] < points[:, 0]
    points = points[b0]

    b1 = bbox[2] > points[:, 0]
    points = points[b1]

    b2 = bbox[1] < points[:, 1]
    points = points[b2]

    b3 = bbox[3] > points[:, 1]
    points = points[b3]

    if labels is not None:
        labels = labels[b0]
        labels = labels[b1]
        labels = labels[b2]
        labels = labels[b3]

    if features is not None:
        features = features[b0]
        features = features[b1]
        features = features[b2]
        features = features[b3]

    return points, labels, features


def normalize(points):
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / m
    return points

