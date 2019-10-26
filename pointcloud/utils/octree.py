"""
My little OcTree Python implementation with some goodies


Coordinate system and quadrants.
    y
3(7)| 0(4)
--------- x
2(6)| 1(5)


Copyright Nejc Dougan 2019
"""
import numpy as np


class OctNode:
    """
    OctNode Class
    """

    def __init__(self, origin, size, data):
        """
        Init OctNode
        :param origin:
        :param size:
        :param data:
        """
        self.origin = np.array(origin)
        self.size = size
        self.data = data
        self.stats = None
        self.is_leaf_node = True
        self.nodes = [None, None, None, None, None, None, None, None]

    def get_data(self):
        """
        Return data
        :return:
        """
        return self.data

    def get_count(self):
        """
        Returns count of points
        :return:
        """
        return np.shape(self.data)[0]

    def get_size(self):
        """
        Returns size of node
        :return:
        """
        return self.size

    def get_stats(self):
        """
        Return Dictionary of statistics
        :return:
        """
        self.stats = {
            "std": np.std(self.data, axis=0),
            "mean": np.mean(self.data, axis=0)
        }
        return self.stats

    def get_children(self):
        """
        Returns all children nodes
        :return:
        """
        return self.nodes

    def get_child(self, octant):
        """
        Returns child at octant
        :param octant:
        :return:
        """
        if octant > 7:
            raise ValueError
        return self.nodes[octant]

    def get_origin(self):
        """
        Returns origin of Node
        :return:
        """
        return self.origin

    def add_child(self, i, origin, size, data):
        """
        Add new child
        :param i:
        :param origin:
        :param size:
        :param data:
        :return:
        """
        if i > 7:
            raise ValueError('Octant Number cannot be larger then 7')

        self.is_leaf_node = False
        self.nodes[i] = OctNode(origin, size, data)

    def clear_node(self):
        """
        Clear nodes data
        :return:
        """
        self.is_leaf_node = False
        self.data = []

    def child_are_leaf_nodes(self):
        """
        Checks if all children are leaf nodes
        :return:
        """
        for node in self.nodes:
            if node is None:
                continue

            if not node.isLeafNode:
                return False

        return True

    def get_all_children_data(self):
        """
        Get data of all children
        :return:
        """
        child_data = None
        for node in self.nodes:
            if node is None:
                continue

            if child_data is None:
                child_data = node.get_data()
            else:
                np.concatenate([child_data, node.getData()])
        return child_data


class OcTree:
    """
    OcTree
    """

    def __init__(self, data):
        """

        :param nodePointLimit:
        :param data:
        """

        # Calculate world size
        dx = np.max(data[:, 0]) - np.min(data[:, 0])
        dy = np.max(data[:, 1]) - np.min(data[:, 1])
        dz = np.max(data[:, 2]) - np.min(data[:, 2])
        world_size = np.ceil(np.max((dx, dy, dz)))

        # Calculates origin
        mx = np.max(data[:, 0]) - dx / 2
        my = np.max(data[:, 1]) - dy / 2
        mz = np.max(data[:, 2]) - dz / 2
        origin = (mx, my, mz)

        self.root = OctNode(origin, world_size, data)

    def grow(self, current_node=None, max_number_of_points=100):
        """
        Grow OcTree
        :param max_number_of_points:
        :param current_node:
        :return:
        """
        if current_node is None:
            current_node = self.root

        if current_node.get_count() < max_number_of_points:
            return True

        current_node = self.divide(current_node)
        children_nodes = current_node.get_children()

        for child_node in children_nodes:
            if child_node is None:
                continue

            self.grow(child_node, max_number_of_points=max_number_of_points)

        return True

    def grow_based_on_stats(self, current_node=None, min_size=0.1, max_size=1, max_points=1000, min_points=6,
                            max_stds=[10, 10, 10, 0.02, 0.02, 0.02]):
        """

        :param current_node:
        :param min_size:
        :param max_size:
        :param max_points:
        :param min_points:
        :param max_stds:
        :return:
        """
        def stop_dividing(current_node_1, min_size_1, max_size_1, max_points_1, min_points_1, max_stds_1):
            """

            :param current_node:
            :param min_size:
            :param max_size:
            :param max_points:
            :param min_points:
            :param max_stds:
            :return:
            """
            size = current_node_1.get_size()
            data = current_node_1.get_data()
            n_points = np.shape(data)[0]

            if n_points > max_points_1 or n_points < min_points_1:
                return False

            if size < min_size_1:
                return True

            if size > max_size_1:
                return False

            stds = np.std(data, axis=0)
            for std_th, std in zip(max_stds_1, stds):
                if std > std_th:
                    return False

            return True

        if current_node is None:
            current_node = self.root

        # If data meets criteria stop dividing it.
        if stop_dividing(current_node, min_size_1=min_size, max_size_1=max_size, max_points_1=max_points,
                         min_points_1=min_points, max_stds_1=max_stds):
            return True

        current_node = self.divide(current_node)
        children_nodes = current_node.get_children()

        for child_node in children_nodes:
            if child_node is None:
                continue

            self.grow_based_on_stats(child_node, min_size=min_size, max_size=max_size, max_points=max_points,
                                     min_points=min_points, max_stds=max_stds)

        return True

    def divide(self, node):
        """
        Divide node into new nodes
        :param node:
        :return:
        """
        node_size = node.get_size()
        node_origin = node.get_origin()
        node_data = node.get_data()
        child_size = node_size * 0.5
        count = 0

        for i, branch in enumerate(node.get_children()):
            child_origin = self.get_position_for_child(i, node_size, node_origin)
            child_data = self.get_data_for_child_node(child_origin, child_size, node_data)

            if np.shape(child_data)[0] == 0:
                continue
            node.add_child(i, child_origin, child_size, child_data)
            count += np.shape(child_data)[0]

        node.clear_node()
        return node

    @staticmethod
    def get_position_for_child(i, size, origin):
        """

        :param i:
        :param size:
        :param origin:
        :return:
        """
        d = size * 0.25

        if i == 0:
            offsets = np.array((d, d, d))
        elif i == 1:
            offsets = np.array((d, -d, d))
        elif i == 2:
            offsets = np.array((-d, -d, d))
        elif i == 3:
            offsets = np.array((-d, d, d))
        elif i == 4:
            offsets = np.array((d, d, -d))
        elif i == 5:
            offsets = np.array((d, -d, -d))
        elif i == 6:
            offsets = np.array((-d, -d, -d))
        elif i == 7:
            offsets = np.array((-d, d, -d))
        else:
            raise Exception

        return origin + offsets

    @staticmethod
    def get_data_for_child_node(origin, size, data):
        """
        Clip data to node BBOX
        :param origin:
        :param size:
        :param data:
        :return:
        """
        half_size = size * 0.5
        xmax = origin[0] + half_size
        xmin = origin[0] - half_size
        ymax = origin[1] + half_size
        ymin = origin[1] - half_size
        zmax = origin[2] + half_size
        zmin = origin[2] - half_size

        data = data[data[:, 0] <= xmax]
        data = data[data[:, 0] > xmin]
        data = data[data[:, 1] <= ymax]
        data = data[data[:, 1] > ymin]
        data = data[data[:, 2] <= zmax]
        data = data[data[:, 2] > zmin]

        return data

    def find_node(self, coordinates, node=None):
        """

        :type node: object
        :param coordinates:
        :return:
        """
        if node is None:
            node = self.root

        if node.is_leaf_node:
            return node

        child_node_octant = self.find_octant(node.get_origin(), coordinates)
        child_node = node.get_child(child_node_octant)

        if child_node is None:
            return child_node

        self.find_node(coordinates, node=child_node)

    @staticmethod
    def find_octant(node_origin, search_coordinates):
        """
        Find right octant
        :param node_origin:
        :param search_coordinates:
        :return:
        """

        result = 0
        mapping = {7: 0, 5: 1, 4: 2, 6: 3, 3: 4, 1: 5, 0: 6, 2: 7}
        for i in range(2):
            if node_origin[i] <= search_coordinates[i]:
                result += 1 + (1 * i)
        if node_origin[2] <= search_coordinates[2]:
            result += 4
        return mapping[result]

    def walk_leaf_nodes(self, node=None):
        """
        Returns Iterator that yield all leaf node
        :param node:
        :return: Iterator
        """
        if node is None:
            node = self.root

        for child in node.get_children():
            if child is None:
                continue

            if child.is_leaf_node:
                yield (child)
            else:
                yield from self.walk_leaf_nodes(child)

        return None

    def walk_leaf_node_parents(self, node=None):
        """
        Returns Iterator that yield all node where all childes are leaf nodes
        :param node:
        :return:
        """
        if node is None:
            node = self.root

        if node.childs_are_leaf_nodes():
            yield (node)
        else:
            for child in node.get_children():
                if child is None:
                    continue

                yield from self.walk_leaf_node_parents(child)

        return None

    def get_number_of_leaf_nodes(self):
        """
        Returns the number of leaf nodes in OcTree
        :return:
        """
        walk = self.walk_leaf_nodes()
        n = 0
        for _ in walk:
            n += 1
        return n

