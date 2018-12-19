import laspy
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
# from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def filterPointsByClass(points, classification):
    return points[points[:, 3] == classification]


def clusterPoints(points):
    neigh = NearestNeighbors(radius=5)
    neigh.fit(points)
    A = neigh.radius_neighbors_graph(points)

    db = DBSCAN(metric='precomputed', eps=1, min_samples=1000).fit(A)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    print(labels, len(labels), points[0])
    return 0


def getAllUniqueClasses(points):
    return np.unique(points[:, 3])


def show(points):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # for c, m, zlow, zhigh in [('r', 'o', -50, -25), ('b', '^', -30, -5)]:
    # xs = randrange(n, 23, 32)
    # ys = randrange(n, 0, 100)
    # zs = randrange(n, zlow, zhigh)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    plt.show()


def process(path):
    point_cloud = laspy.file.File(path, mode='r')
    points = np.vstack((point_cloud.x, point_cloud.y, point_cloud.z, point_cloud.classification)).transpose()
    all_classes = getAllUniqueClasses(points)

    # for classification in all_classes:
    #     filtered_point_cloud = filterPointsByClass(points, classification)
    #     clusterPoints(filtered_point_cloud)

    filtered_point_cloud = filterPointsByClass(points, 16)
    show(filtered_point_cloud)


if __name__ == "__main__":
    FOLDER = '../../../datasets/flycom/kellag/'
    FILE_NAME = 'kelag000001.las'
    print('Reading point cloud {0}'.format(FILE_NAME))
    process(FOLDER + FILE_NAME)
