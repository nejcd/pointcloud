"""
Yeah my ocrtee program

Coordinate system and quadrants. Below origin are in paranthes
    y
3(7)| 0(4)
--------- x
2(6)| 1(5)



Copyright Nejc Dougan 2019
"""


class OctNode:
    def __init__(self, origin, size, data):
        self.origin = np.array(origin)
        self.size = size
        self.data = data
        self.isLeafNode = True
        self.nodes = [None, None, None, None, None, None, None, None]

    def getData(self):
        return self.data

    def getCount(self):
        return np.shape(self.data)[0]

    def getSize(self):
        return self.size

    def getChildren(self):
        return self.nodes

    def getChild(self, octant):
        return self.nodes[octant]

    def getOrigin(self):
        return self.origin

    def addChild(self, i, origin, size, data):
        self.nodes[i] = OctNode(origin, size, data)

    def clearNode(self):
        self.isLeafNode = False
        self.data = []


class OcTree:

    def __init__(self, worldSize, nodePointLimit, origin=(0, 0, 0), data=[]):
        self.nodePointLimit = nodePointLimit
        self.root = OctNode(origin, worldSize, data)

    def grow(self, currentNode=None):
        if currentNode == None:
            currentNode = self.root

        if (currentNode.getCount() < self.nodePointLimit):
            return True

        currentNode = self.divide(currentNode)
        childrenNodes = currentNode.getChildren()

        for childNode in childrenNodes:
            if childNode is None:
                continue

            self.grow(childNode)

        return True

    def divide(self, node):
        nodeSize = node.getSize()
        nodeOrigin = node.getOrigin()
        nodeData = node.getData()
        childSize = nodeSize * 0.5
        count = 0;

        for i, brench in enumerate(node.getChildren()):
            childOrigin = self.getPositionForChild(i, nodeSize, nodeOrigin)
            childData = self.getDataForChildNode(childOrigin, childSize, nodeData)
            node.addChild(i, childOrigin, childSize, childData)
            count += np.shape(childData)[0]

        node.clearNode()
        return node

    def getPositionForChild(self, i, size, origin):
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

    def getDataForChildNode(self, origin, size, data):
        halfsize = size * 0.5
        xmax = origin[0] + halfsize
        xmin = origin[0] - halfsize
        ymax = origin[1] + halfsize
        ymin = origin[1] - halfsize
        zmax = origin[2] + halfsize
        zmin = origin[2] - halfsize

        data = data[data[:, 0] <= xmax]
        data = data[data[:, 0] > xmin]
        data = data[data[:, 1] <= ymax]
        data = data[data[:, 1] > ymin]
        data = data[data[:, 2] <= zmax]
        data = data[data[:, 2] > zmin]

        return data

    def findNode(self, coordinates, node=None):
        if node is None:
            node = self.root

        if node.isLeafNode:
            return node

        childNodeOctant = self.findOctant(node.getOrigin(), coordinates)
        childNode = node.getChild(childNodeOctant)

        if childNode is None:
            return childNode

        self.findNode(coordinates, childNode)

    def findOctant(self, node_origin, search_coordinates):
        result = 0
        mapping = {7: 0, 5: 1, 4: 2, 6: 3, 3: 4, 1: 5, 0: 6, 2: 7}
        for i in range(2):
            if node_origin[i] <= search_coordinates[i]:
                result += 1 + (1 * i)
        if node_origin[2] <= search_coordinates[2]:
            result += 4
        return mapping[result]

