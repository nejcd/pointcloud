import numpy as np


def by_label(pointcloud, labels, label):
    """

    :param pointcloud:
    :param labels:
    :param label:
    :return:
    """
    return pointcloud[labels == label]