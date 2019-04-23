import glob
import math

import matplotlib.pyplot as plt
import numpy as np
from descartes.patch import PolygonPatch
from matplotlib import pyplot
from plyfile import PlyData, PlyElement
from shapely.geometry import Polygon, MultiPolygon

from pointcloud.utils import processing, readers
from pointcloud.utils.eulerangles import euler2mat


def calculate_polygon_from_filename(file_name, grid_size, x_pos, y_pos):
    """
    :param file_name:
    :param grid_size:
    :param x_pos:
    :param y_pos:
    :return:
    """
    x_min = 0
    y_min = 0

    try:
        exploded = file_name.split('.')[0].split('_')
        x_min = int(exploded[x_pos])
        y_min = int(exploded[y_pos])
    except IndexError:
        print('Could not parse Filename for Polygon, check if x na y pos correct')

    return Polygon([(x_min, y_min),
                    (x_min + grid_size, y_min),
                    (x_min + grid_size, y_min + grid_size),
                    (x_min, y_min + grid_size)])


def get_names_and_polygons_in_workspace(workspace, settings=None, polygon_from_filename_settings=None,
                                        file_format='las', file_format_settings=None):
    """
    :param file_format:
    :param workspace: Path to workspace
    :param settings: Dictionary keys: step, x_pos, y_pos
    :param polygon_from_filename_settings:
    :return:
    """
    if file_format == 'las':
        reader = readers.LasReader()
    elif file_format == 'txt':
        reader = readers.TxtReader(xyz=file_format_settings['xyz'],
                                   label=file_format_settings['label'],
                                   features=file_format_settings['features'])
    elif file_format == 'npy':
        reader = readers.NpyReader(xyz=file_format_settings['xyz'],
                                   label=file_format_settings['label'],
                                   features=file_format_settings['features'])
    else:
        raise Exception('Not supported file format')
    files = glob.glob(workspace + "/*." + reader.extension)

    out = []
    if len(files) == 0:
        raise UserWarning('No files in current workspace')
    for file in files:
        filename = file.split('/')[-1]
        filename = filename.split('.')[0]
        if polygon_from_filename_settings is not None:
            step, x_pos, y_pos = get_polygon_from_file_settings(settings)
            polygon = calculate_polygon_from_filename(filename, step, x_pos, y_pos)
        else:
            points = reader.get_points(workspace + filename)
            polygon = processing.boundary(points)
        out.append({'name': filename, 'polygon': polygon})

    return out


def get_polygon_from_file_settings(settings):
    """
    :param settings:
    :return:
    """
    try:
        return settings['step'], settings['x_pos'], settings['y_pos']
    except ValueError:
        print('Not Valid Settings')


def calculate_tile_size_from_target_number_of_points(num_points, density, tile_type='grid'):
    """
    :param num_points:
    :param density:
    :param tile_type:
    :return:
    """
    target_area_size = num_points / density

    if tile_type == 'grid':
        return calculate_grid_dimensions(target_area_size)
    elif tile_type == 'circle':
        return calculate_circle_dimensions(target_area_size)
    else:
        raise ValueError('Tile type {0} not supported (grid, circle)'.format(tile_type))


def calculate_grid_dimensions(area):
    """
    :param area:
    :return:
    """
    return round(math.sqrt(area), 0)


def calculate_circle_dimensions(area):
    """
    :param area:
    :return:
    """
    return round(math.sqrt(area / math.pi))


def create_fish_net(bbox, size):
    """
    Generate Fish net
    :param bbox:
    :param size:
    :return:
    """
    min_x = bbox[0]
    min_y = bbox[1]
    dx = bbox[2] - min_x
    dy = bbox[3] - min_y
    steps_x = int(math.ceil(dx / size))
    steps_y = int(math.ceil(dy / size))

    polygons = [None] * (steps_x * steps_y)
    n = 0

    for i in range(steps_x):
        for j in range(steps_y):
            polygons[n] = Polygon([
                (min_x + (i * size), min_y + (j * size)),
                (min_x + ((i + 1) * size), min_y + (j * size)),
                (min_x + ((i + 1) * size), min_y + ((j + 1) * size)),
                (min_x + (i * size), min_y + ((j + 1) * size)),
            ])
            n += 1

    return MultiPolygon([polygon for polygon in polygons])


# ----------------------------------------
# Point Cloud/Volume Conversions
# ----------------------------------------

def point_cloud_to_volume_batch(point_clouds, vsize=12, radius=1.0, flatten=True):
    """ Input is BxNx3 batch of point cloud
        Output is Bx(vsize^3)
    """
    vol_list = []
    for b in range(point_clouds.shape[0]):
        vol = point_cloud_to_volume(np.squeeze(point_clouds[b, :, :]), vsize, radius)
        if flatten:
            vol_list.append(vol.flatten())
        else:
            vol_list.append(np.expand_dims(np.expand_dims(vol, -1), 0))
    if flatten:
        return np.vstack(vol_list)
    else:
        return np.concatenate(vol_list, 0)


def point_cloud_to_volume(points, vsize, radius=1.0):
    """ input is Nx3 points.
        output is vsize*vsize*vsize
        assumes points are in range [-radius, radius]
    """
    vol = np.zeros((vsize, vsize, vsize))
    voxel = 2 * radius / float(vsize)
    locations = (points + radius) / voxel
    locations = locations.astype(int)
    vol[locations[:, 0], locations[:, 1], locations[:, 2]] = 1.0
    return vol


def volume_to_point_cloud(vol):
    """ vol is occupancy grid (value = 0 or 1) of size vsize*vsize*vsize
        return Nx3 numpy array.
    """
    vsize = vol.shape[0]
    assert (vol.shape[1] == vsize and vol.shape[1] == vsize)
    points = []
    for a in range(vsize):
        for b in range(vsize):
            for c in range(vsize):
                if vol[a, b, c] == 1:
                    points.append(np.array([a, b, c]))
    if len(points) == 0:
        return np.zeros((0, 3))
    points = np.vstack(points)
    return points


# ----------------------------------------
# Point cloud IO
# ----------------------------------------

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z] for x, y, z in pc])
    return pc_array


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)


def draw_point_cloud(input_points, canvasSize=500, space=200, diameter=25,
                     xrot=0, yrot=0, zrot=0, switch_xyz=[0, 1, 2], normalize=True):
    """ Render point cloud to image with alpha channel.
        Input:
            points: Nx3 numpy array (+y is up direction)
        Output:
            gray image as numpy array of size canvasSizexcanvasSize
    """
    image = np.zeros((canvasSize, canvasSize))
    if input_points is None or input_points.shape[0] == 0:
        return image

    points = input_points[:, switch_xyz]
    M = euler2mat(zrot, yrot, xrot)
    points = (np.dot(M, points.transpose())).transpose()

    # Normalize the point cloud
    # We normalize scale to fit points in a unit sphere
    if normalize:
        centroid = np.mean(points, axis=0)
        points -= centroid
        furthest_distance = np.max(np.sqrt(np.sum(abs(points) ** 2, axis=-1)))
        points /= furthest_distance

    # Pre-compute the Gaussian disk
    radius = (diameter - 1) / 2.0
    disk = np.zeros((diameter, diameter))
    for i in range(diameter):
        for j in range(diameter):
            if (i - radius) * (i - radius) + (j - radius) * (j - radius) <= radius * radius:
                disk[i, j] = np.exp((-(i - radius) ** 2 - (j - radius) ** 2) / (radius ** 2))
    mask = np.argwhere(disk > 0)
    dx = mask[:, 0]
    dy = mask[:, 1]
    dv = disk[disk > 0]

    # Order points by z-buffer
    zorder = np.argsort(points[:, 2])
    points = points[zorder, :]
    points[:, 2] = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2] - np.min(points[:, 2])))
    max_depth = np.max(points[:, 2])

    for i in range(points.shape[0]):
        j = points.shape[0] - i - 1
        x = points[j, 0]
        y = points[j, 1]
        xc = canvasSize / 2 + (x * space)
        yc = canvasSize / 2 + (y * space)
        xc = int(np.round(xc))
        yc = int(np.round(yc))

        px = dx + xc
        py = dy + yc

        image[px, py] = image[px, py] * 0.7 + dv * (max_depth - points[j, 2]) * 0.3

    image = image / np.max(image)
    return image


# -----------------------
# Points Visualization
# -----------------------


def point_cloud_three_views(points):
    """ input points Nx3 numpy array (+y is up direction).
        return an numpy array gray image of size 500x1500. """
    img1 = draw_point_cloud(points, zrot=110 / 180.0 * np.pi, xrot=45 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img2 = draw_point_cloud(points, zrot=70 / 180.0 * np.pi, xrot=135 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    img3 = draw_point_cloud(points, zrot=180.0 / 180.0 * np.pi, xrot=90 / 180.0 * np.pi, yrot=0 / 180.0 * np.pi)
    image_large = np.concatenate([img1, img2, img3], 1)
    return image_large


def pyplot_draw_point_cloud(points):
    """ points is a Nx3 numpy array """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    # savefig(output_filename)


def pyplot_draw_volume(vol, output_filename):
    """ vol is of size vsize*vsize*vsize
        output an image to output_filename
    """
    points = volume_to_point_cloud(vol)
    pyplot_draw_point_cloud(points)


def plot_3d(points, max_points=1000, title=None, save=False, path=None, labels=None, label_vector=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    color = 'k'
    label = 'point'

    np.random.shuffle(points)
    plotpoints = points[:max_points, :]

    if labels is not None:
        for key, l in labels.items():
            group = []
            if label_vector:
                point_labels = label_vector
            else:
                point_labels = plotpoints[:, 3]
            group = plotpoints[point_labels == int(key)]
            color = l['color']
            label = l['name']
            if len(group) > 0:
                ax.scatter(group[:, 0], group[:, 1], group[:, 2], c=color, label=label)

    else:
        for point in plotpoints:
            code = 0
            if len(point) >= 4:
                color = 'k'
                label = str(int(point[3]))

            ax.scatter(point[0], point[1], point[2], c=color)

    if title:
        ax.set_title(title)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc='upper left')
    plt.axis('equal')

    if save:
        if path is None:
            raise Exception('Path to location has to be provided')
        plt.savefig('{0}plt_{1}.png'.format(path, title))
    else:
        plt.show()


# -----------------------
# Project Visualization
# -----------------------

def plot_polygons(multipolygons, title=None):
    """
    :param title:
    :param multipolygons:
    :return:
    """

    fig = pyplot.figure(1, dpi=90)
    ax = fig.add_subplot(121)

    if title:
        ax.set_title(title)

    for polygon in multipolygons:
        patch = PolygonPatch(polygon, facecolor='#6699cc', edgecolor='#ffffff', alpha=0.5, zorder=2)
        ax.add_patch(patch)

    pyplot.show()
