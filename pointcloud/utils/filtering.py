import numpy as np


def by_label(pointcloud, label):
    labels = pointcloud[:, 3]
    return pointcloud[labels == label]


if __name__ == '__main__':
    test_pc = np.array([[1, 1, 1, 0],
                        [1, 2, 1, 0],
                        [3, 1, 1, 0],
                        [4, 5, 1, 0],
                        [3, 6, 10, 1],
                        [2, 5, 10, 1],
                        [4, 6, 10, 1],
                        [3, 5, 10, 1]])

    filteterd = by_label(test_pc, 0)