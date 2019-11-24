import functools
from pathlib import Path

import laspy
import numpy as np

_cache_size = 2048


class LasReader(object):

    def __init__(self, path=None, settings=None, extension='las'):
        """

        :param path:
        :param settings:
        """
        self.extension = extension

        self.path = path

        if path is not None:
            filename = Path(path)
            if filename.suffix == '.' + self.extension:
                self.filename = filename
            else:
                self.filename = filename.with_suffix('.' + self.extension)

        self.points = 'scaled'
        self.labels = None
        self.features = None

        if settings is not None:
            self.points = settings['points']
            self.features = settings['features']
            self.labels = settings['labels']

    def get_extension(self):
        return self.extension

    # @functools.lru_cache(maxsize=_cache_size)
    def load_data(self, path):
        """
        :return:
        """
        with laspy.file.File(path, mode='r') as file:
            if self.points == 'scaled':
                points = np.vstack((file.x, file.y, file.z)).transpose()
            else:
                points = np.vstack((file.X, file.Y, file.Z)).transpose()

            labels = np.vstack(file.classification)
            out_features = None
            for feature in self.features:
                f = None
                if feature == 'intensity':
                    f = np.vstack(file.intensity).transpose()
                elif feature == 'num_return':
                    f = np.vstack(file.num_returns).transpose()
                elif feature == 'return_num':
                    f = np.vstack(file.return_num).transpose()
                elif feature == 'rgb':
                    f = np.vstack((file.red, file.green, file.blue)).transpose()
                elif feature == 'RGB':
                    f = np.vstack((file.Red, file.Green, file.Blue)).transpose()

                if f is not None:
                    if out_features is None:
                        out_features = f
                    else:
                        out_features = np.concatenate((out_features, f), axis=0)

        return points, np.array(np.squeeze(labels), dtype=np.int32), out_features.transpose()

    def get_all(self, path=None):
        """
        Returns points, labels and features
        :return: points, labels, features
        """
        if path is None:
            path = self.filename

        return self.load_data(path)

    def get_points(self, path=None):
        """
        :return:
        """
        if path is None:
            path = self.filename

        points, _, _ = self.load_data(path)
        return points

    def get_labels(self, path=None):
        """
        :return:
        """
        if path is None:
            path = self.path

        _, labels, _ = self.load_data(path)
        return labels

    def get_features(self, path=None):
        """
        :return:
        """
        if path is None:
            path = self.filename

        _, _, features = self.load_data(path)
        return features

    def store(self, points, labels=None, features=None, path=None):
        """
        :param path:
        :param points:
        :param labels:
        :param features:
        :return:
        """
        header = laspy.header.Header()
        if path is None:
            path = self.filename

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

    def __init__(self, path=None, settings=None, extension='txt'):
        """

        :param path:
        :param settings:
        :param extension:
        """
        self.extension = extension
        self.path = path

        filename = Path(path)
        if filename.suffix == '.' + self.extension:
            self.filename = filename
        else:
            self.filename = filename.with_suffix('.' + self.extension)

        self.points = [0, 1, 2]
        self.features = [3, 4, 5]
        self.labels = [6]
        if settings is not None:
            self.points = settings['points']
            self.features = settings['features']
            self.labels = settings['labels']

    def get_extension(self):
        return self.extension

    @functools.lru_cache(maxsize=_cache_size)
    def load_data(self, path):
        """
        :param path:
        :return:
        """
        return np.loadtxt(path)

    def get_all(self, path=None):
        """

        :return:
        """
        if path is None:
            path = self.path

        return self.get_points(path), self.get_labels(path), self.get_features(path)

    def get_points(self, path=None):
        """
        :param path:
        :return:
        """
        if path is None:
            path = self.path
        data = self.load_data(path)
        points = np.vstack((data[:, self.points[0]], data[:, self.points[1]], data[:, self.points[2]])).transpose()
        return points

    def get_labels(self, path=None):
        """
        :return:
        """
        if self.labels is None:
            raise Exception('Labels not set')
        if path is None:
            path = self.path
        data = self.load_data(path)
        labels = data[:, self.labels[0]]
        return labels

    def get_features(self, path=None):
        """
        :return:
        """
        if self.features is None:
            raise Exception('Features not set')

        if path is None:
            path = self.path

        data = self.load_data(self.path)
        features = None
        for feature in self.features:
            if features is None:
                features = data[:, feature]
            else:
                features = np.vstack((features, data[:, feature]))

        return features.transpose()

    def store(self, points, labels=None, features=None):
        """
        :param points:
        :param labels:
        :param features:
        :return:
        """

        data = points

        if features is not None:
            data = np.hstack((data, features))

        if labels is not None:
            data = np.hstack((data, np.array([labels]).transpose()))

        np.savetxt(self.path, data)


class NpyReader(object):

    def __init__(self, path=None, settings=None, extension='npy'):
        """

        :param path:
        :param settings:
        """
        self.extension = extension
        self.path = path

        filename = Path(path)
        if filename.suffix == '.' + self.extension:
            self.filename = filename
        else:
            self.filename = filename.with_suffix('.' + self.extension)
        self.points = [0, 1, 2]
        self.features = [3, 4, 5]
        self.labels = [6]
        if settings is not None:
            self.points = settings['points']
            self.features = settings['features']
            self.labels = settings['labels']

    def get_extension(self):
        return self.extension

    @functools.lru_cache(maxsize=_cache_size)
    def load_data(self, path):
        """
        :param path:
        :return:
        """
        data = np.load(path)

        if self.extension == 'npz':
            data = data['pc']

        return data

    def get_all(self, path=None):
        """

        :return:
        """
        if path is None:
            path = self.path
        return self.get_points(path), self.get_labels(path), self.get_features(path)

    def get_points(self, path=None):
        """
        :return:
        """
        if path is None:
            path = self.path
        data = self.load_data(path)
        points = np.vstack((data[:, self.points[0]], data[:, self.points[1]], data[:, self.points[2]])).transpose()
        return points

    def get_labels(self, path=None):
        """
        :return:
        """
        if self.labels is None:
            raise Exception('Labels not set')
        if path is None:
            path = self.path
        data = self.load_data(path)
        labels = data[:, self.labels[0]]
        return labels

    def get_features(self, path=None):
        """
        :return:
        """
        if self.features is None:
            raise Exception('Features not set')
        if path is None:
            path = self.path
        data = self.load_data(path)
        features = None
        for feature in self.features:
            if features is None:
                features = data[:, feature]
            else:
                features = np.vstack((features, data[:, feature]))

        return features.transpose()

    def store(self, points, labels=None, features=None):
        """
        :param points:
        :param labels:
        :param features:
        :return:
        """
        data = points

        if features is not None:
            data = np.concatenate((data, features), axis=1)

        if labels is not None:
            data = np.concatenate((data, np.reshape(labels, (np.shape(labels)[0], 1))), axis=1)

        if self.extension == 'npz':
            np.savez_compressed(self.path, pc=data)
        else:
            np.save(self.path, data)
