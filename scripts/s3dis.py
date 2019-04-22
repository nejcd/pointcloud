import numpy as np
from pointcloud.utils import readers


def create_text_file():
    reader = readers.LasReader()
    points = reader.get_points('../tests/test_data/test_tile_27620_158060.las')
    labels = np.ones((np.shape(points)[0], 1))
    features = np.ones((np.shape(points)[0], 3)) * [4, 5, 7]
    out = np.hstack((points, labels, features))
    np.save('../tests/test_data/test_tile_27620_158060.npy', out)


if __name__ == "__main__":
    create_text_file()
