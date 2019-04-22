import laspy
import numpy as np


class LasReader(object):
    extension = 'las'

    def get_points(self, path):
        """
        :param filepath:
        :return:
        """
        path = self.validate_filename(path)
        point_file = laspy.file.File(path, mode='r')
        points = np.vstack((point_file.x, point_file.y, point_file.z)).transpose()
        point_file.close()
        return points

    def get_labels(self, path):
        """
        :return:
        """
        path = self.validate_filename(path)
        point_file = laspy.file.File(path, mode='r')
        labels = np.vstack(point_file.classification)
        point_file.close()
        return labels

    def validate_filename(self, filename):
        """
        :param filename:
        :return:
        """
        if filename.split('.')[-1] == self.extension:
            return filename
        return '{0}.{1}'.format(filename, self.extension)

    def store(self, path, points, labels=None, features=None):
        """
        :param path:
        :param points:
        :param labels:
        :param features:
        :return:
        """
        path = self.validate_filename(path)
        header = laspy.header.Header()

        file_out = laspy.file.File(path, mode='w', header=header)
        file_out.X = np.ndarray.astype(points[:, 0] * 100,
                                       dtype=int)  # TODO DO NICER :) ALSO HANDLE WHOLE HEADER STUFF ETC!
        file_out.Y = np.ndarray.astype(points[:, 1] * 100, dtype=int)
        file_out.Z = np.ndarray.astype(points[:, 2] * 100, dtype=int)

        if labels:
            file_out.classification = np.ndarray.astype(labels, dtype=int)

        # TODO THIS SHOULD GO AWAY
        file_out.header.offset = [0, 0, 0]
        file_out.header.scale = [0.01, 0.01, 0.01]

        file_out.close()


class TxtReader(object):
    extension = 'txt'

    def __init__(self, xyz, label=None, features=None):
        """

        :param xyz: Array of indices where xyz coordinates are located in text file
        :param label: Array of index where label is located in text file
        :param features: Array of indices where features are stored in text file
        """
        self.xyz = xyz
        self.features = features
        self.label = label

    def load_data(self, path):
        """
        :param path:
        :return:
        """
        path = self.validate_filename(path)

        return np.loadtxt(path)

    def get_points(self, path):
        """
        :param path:
        :return:
        """
        data = self.load_data(path)
        points = np.vstack((data[:, self.xyz[0]], data[:, self.xyz[1]], data[:, self.xyz[2]])).transpose()
        return points

    def get_labels(self, path):
        """
        :return:
        """
        data = self.load_data(path)
        labels = data[:, self.label[0]]
        return labels

    def get_features(self, path):
        """
        :return:
        """
        data = self.load_data(path)
        features = None
        for feature in self.features:
            if features is None:
                features = data[:, feature]
            else:
                features = np.vstack((features, data[:, feature]))

        return features.transpose()

    def validate_filename(self, filename):
        """
        :param filename:
        :return:
        """
        if filename.split('.')[-1] == self.extension:
            return filename
        return '{0}.{1}'.format(filename, self.extension)

    def store(self, path, points, labels=None, features=None):
        """
        :param path:
        :param points:
        :param labels:
        :param features:
        :return:
        """
        path = self.validate_filename(path)

        if labels:
            points = np.hstack((points, labels))

        if features:
            points = np.hstack((points, features))

        np.savetxt(path, points)


class NpyReader(object):
    extension = 'npy'

    def __init__(self, xyz, label=None, features=None):
        """

        :param xyz: Array of indices where xyz coordinates are located in text file
        :param label: Array of index where label is located in text file
        :param features: Array of indices where features are stored in text file
        """
        self.xyz = xyz
        self.features = features
        self.label = label

    def load_data(self, path):
        """
        :param path:
        :return:
        """
        path = self.validate_filename(path)

        return np.load(path)

    def get_points(self, path):
        """
        :param path:
        :return:
        """
        data = self.load_data(path)
        points = np.vstack((data[:, self.xyz[0]], data[:, self.xyz[1]], data[:, self.xyz[2]])).transpose()
        return points

    def get_labels(self, path):
        """
        :return:
        """
        data = self.load_data(path)
        labels = data[:, self.label[0]]
        return labels

    def get_features(self, path):
        """
        :return:
        """
        data = self.load_data(path)
        features = None
        for feature in self.features:
            if features is None:
                features = data[:, feature]
            else:
                features = np.vstack((features, data[:, feature]))

        return features.transpose()

    def validate_filename(self, filename):
        """
        :param filename:
        :return:
        """
        if filename.split('.')[-1] == self.extension:
            return filename
        return '{0}.{1}'.format(filename, self.extension)

    def store(self, path, points, labels=None, features=None):
        """
        :param path:
        :param points:
        :param labels:
        :param features:
        :return:
        """
        path = self.validate_filename(path)

        if labels:
            points = np.hstack((points, labels))

        if features:
            points = np.hstack((points, features))

        np.save(path, points)
