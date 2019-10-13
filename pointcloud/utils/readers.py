import laspy
import numpy as np
from pathlib import Path


class LasReader(object):
    extension = 'las'

    def __init__(self, settings=None):
        """

        :param points: Array of indices where xyz coordinates are located in text file
        :param labels: Array of index where label is located in text file
        :param features: Array of indices where features are stored in text file
        """
        self.points = 'scaled'
        self.labels = None
        self.features = None
        if settings is not None:
            self.set_file_format_settings(settings)

    def get_all(self, path):
        """
        Returns points, labelsand features
        :param path:
        :return: points, labels, features
        """
        return self.get_points(path), self.get_labels(path), self.get_features(path)

    def set_file_format_settings(self, settings):
        self.points = settings['points']
        self.features = settings['features']
        self.labels = settings['labels']

    def get_points(self, path):
        """
        :return:
        """
        path = self.validate_filename(path)
        point_file = laspy.file.File(path, mode='r')
        if self.points == 'scaled':
            points = np.vstack((point_file.x, point_file.y, point_file.z)).transpose()
        else:
            points = np.vstack((point_file.X, point_file.Y, point_file.Z)).transpose()
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

    def get_features(self, path):
        """
        :return:
        """
        path = self.validate_filename(path)
        point_file = laspy.file.File(path, mode='r')

        features = None
        for feature in self.features:
            f = None
            if feature == 'intensity':
                f =  np.vstack(point_file.intensity).transpose()
            elif feature == 'num_return':
                f =  np.vstack(point_file.num_returns).transpose()
            elif feature == 'return_num':
                f =  np.vstack(point_file.return_num).transpose()
            elif feature == 'rgb':
                f = np.vstack((point_file.red, point_file.green, point_file.blue)).transpose()
            elif feature == 'RGB':
                f = np.vstack((point_file.Red, point_file.Green, point_file.Blue)).transpose()

            if f is not None:
                if features is None:
                    features = f
                else:
                    features = np.concatenate((features, f), axis=0)

        point_file.close()
        return features.transpose()

    def validate_filename(self, filename):
        """
        :param filename:
        :return:
        """
        filename = Path(filename)
        if filename.suffix == '.' + self.extension:
            return filename
        return filename.with_suffix('.' + self.extension)

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

        if labels is not None:
            file_out.classification = np.ndarray.astype(labels, dtype=int)

        if features is not None:
            raise Warning('Not implemented. (Features will not be saved)')

        # TODO THIS SHOULD GO AWAY
        file_out.header.offset = [0, 0, 0]
        file_out.header.scale = [0.01, 0.01, 0.01]

        file_out.close()


class TxtReader(object):

    def __init__(self, settings=None, extension='txt'):
        """

        :param points: Array of indices where xyz coordinates are located in text file
        :param labels: Array of index where label is located in text file
        :param features: Array of indices where features are stored in text file
        """
        self.extension = extension
        self.points = [0, 1, 2]
        self.features = [3, 4, 5]
        self.labels = [6]
        if settings is not None:
            self.set_file_format_settings(settings=settings)

    def set_file_format_settings(self, settings):
        self.points = settings['points']
        self.features = settings['features']
        self.labels = settings['labels']

    def load_data(self, path):
        """
        :param path:
        :return:
        """
        path = self.validate_filename(path)

        return np.loadtxt(path)

    def get_all(self, path):
        """

        :param path:
        :return:
        """
        data = self.load_data(path)
        points = np.vstack((data[:, self.points[0]], data[:, self.points[1]], data[:, self.points[2]])).transpose()

        if self.labels is None:
            labels = None
        else:
            labels = data[:, self.labels[0]]

        if self.features is None:
            features = None
        else:
            features = []
            for feature in self.features:
                if features is None:
                    features = data[:, feature]
                else:
                    features = np.vstack((features, data[:, feature]))
            features = features.transpose()

        return points, labels, features

    def get_points(self, path):
        """
        :param path:
        :return:
        """
        data = self.load_data(path)
        points = np.vstack((data[:, self.points[0]], data[:, self.points[1]], data[:, self.points[2]])).transpose()
        return points

    def get_labels(self, path):
        """
        :return:
        """
        if self.labels is None:
            raise Exception('Labels not set')

        data = self.load_data(path)
        labels = data[:, self.labels[0]]
        return labels

    def get_features(self, path):
        """
        :return:
        """
        if self.features is None:
            raise Exception('Features not set')

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
        filename = Path(filename)
        if filename.suffix == '.' + self.extension:
            return filename
        return filename.with_suffix('.' + self.extension)

    def store(self, path, points, labels=None, features=None):
        """
        :param path:
        :param points:
        :param labels:
        :param features:
        :return:
        """
        path = self.validate_filename(path)

        data = points

        if features is not None:
            data = np.hstack((data, features))

        if labels is not None:
            data = np.hstack((data, np.array([labels]).transpose()))

        np.savetxt(path, data)


class NpyReader(object):
    extension = 'npy'

    def __init__(self, settings=None):
        """
        :param points: Array of indices where xyz coordinates are located in text file
        :param labels: Array of index where label is located in text file
        :param features: Array of indices where features are stored in text file
        """
        self.points = [0, 1, 2]
        self.features = [3, 4, 5]
        self.labels = [6]
        if settings is not None:
            self.set_file_format_settings(settings)


    def load_data(self, path):
        """
        :param path:
        :return:
        """
        path = self.validate_filename(path)

        return np.load(path)

    def set_file_format_settings(self, settings):
        self.points = settings['points']
        self.features = settings['features']
        self.labels = settings['labels']

    def get_all(self, path):
        """

        :param path:
        :return:
        """
        data = self.load_data(path)
        points = np.vstack((data[:, self.points[0]], data[:, self.points[1]], data[:, self.points[2]])).transpose()

        if self.labels is None:
            labels = None
        else:
            labels = data[:, self.labels[0]]

        if self.features is None:
            features = None
        else:
            features = []
            for feature in self.features:
                if features is None:
                    features = data[:, feature]
                else:
                    features = np.vstack((features, data[:, feature]))
            features = features.transpose()

        return points, labels, features


    def get_points(self, path):
        """
        :param path:
        :return:
        """
        data = self.load_data(path)
        points = np.vstack((data[:, self.points[0]], data[:, self.points[1]], data[:, self.points[2]])).transpose()
        return points

    def get_labels(self, path):
        """
        :return:
        """
        if self.labels is None:
            raise Exception('Labels not set')

        data = self.load_data(path)
        labels = data[:, self.labels[0]]
        return labels

    def get_features(self, path):
        """
        :return:
        """
        if self.features is None:
            raise Exception('Features not set')

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
        filename = Path(filename)
        if filename.suffix == '.' + self.extension:
            return filename
        return filename.with_suffix('.' + self.extension)

    def store(self, path, points, labels=None, features=None):
        """
        :param path:
        :param points:
        :param labels:
        :param features:
        :return:
        """
        path = self.validate_filename(path)

        data = points

        if features is not None:
            data = np.hstack((data, features))

        if labels is not None:
            data = np.hstack((data, np.array([labels]).transpose()))

        np.save(path, data)
