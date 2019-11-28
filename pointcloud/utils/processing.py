import numpy as np
from scipy.spatial import ConvexHull, Delaunay, kdtree
import matplotlib.pyplot as plt
from shapely.geometry import Polygon


def sample_to_target_size(points, target_size, shuffle=True, labels=None, features=None, seed=None):
    """
    Return N (target_size) of data and shuffles it
    Beware original NP Array Gets Shuffled Also!!!!

    :param features:
    :param seed:
    :param labels:
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

    pout = (points[:target_size, :])

    if labels is not None:
        lout = (labels[:target_size])
    else:
        lout = None

    if features is not None:
        fout = (features[:target_size])
    else:
        fout = None

    return pout, lout, fout


def remove_label(points, labels, label, features=None):
    """
    :param points:
    :param features:
    :param labels:
    :param label:
    :return:
    """
    remove = [labels != label]

    if features is not None:
        fout = features[remove]
    else:
        fout = None

    return points[remove], labels[remove], fout


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
    """
    Points getting mean centered and normalized between -1 and 1
    :param points:
    :return:
    """
    centroid = np.mean(points, axis=0)
    points = points - centroid
    m = np.max(np.sqrt(np.sum(points ** 2, axis=1)))
    points = points / m
    return points


def compute_normal_from_points(points):
    """
    Computes normal for a point
    :param points:
    :return:
    """
    epsilon = 10e-9

    delaunay_mesh = Delaunay(points[:, 0:2], qhull_options='QJ')
    tris = points[delaunay_mesh.simplices]  ## TODO should every tris be mean centered ?

    normals = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])

    lens = np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2 + normals[:, 2] ** 2)
    normals[:, 0] /= (lens + epsilon)
    normals[:, 1] /= (lens + epsilon)
    normals[:, 2] /= (lens + epsilon)

    return np.mean(normals, axis=0)


def compute_normals_for_all_points(points, n_size=36):
    """

    :param n_size:
    :param points:
    :return:
    """
    tree = kdtree.KDTree(points)
    normals = []
    for point in points:
        d, i = tree.query(point, k=n_size)
        current_points = points[i] - point
        normals.append(compute_normal_from_points(current_points))

    return np.array(normals)


def classify_close_by(points, labels, from_label, to_label, close_to_label, radius, kdTree=None):
    """

    :param radius:
    :param close_to_label:
    :param to_label:
    :param from_label:
    :param labels:
    :param points:
    :param kdTree:
    :return:
    """
    if kdTree is None:
        kdTree = kdtree.KDTree(points)

    for i, point_label in enumerate(zip(points, labels)):
        point = point_label[0]
        label = point_label[1]

        if label != from_label:
            continue

        i = kdTree.query_ball_point(point, r=radius)
        qpoints = points[i]
        qlabels = labels[i]
        npoints = qpoints[qlabels == close_to_label]

        if len(npoints) == 0:
            continue

        labels[i] = to_label

    return labels


def remap_labels(labels, mappings):
    """

    :param labels:
    :param mappings:
    :return:
    """
    for source, target in mappings.items():
        labels[labels == int(source)] = target

    return labels


def point_count_per_class(labels, number_of_classes=100):
    """

    :return:
    """

    fq = np.zeros(number_of_classes)
    for label in labels:
        fq[label] += 1

    #TODO np.unique(labels, return_counts=True)

    return fq


def test():
    print('Hello from Python')
