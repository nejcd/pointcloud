import multiprocessing as mp
import sys
import gc
import geojson
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import time
from descartes.patch import PolygonPatch
from matplotlib import pyplot
from plyfile import PlyData, PlyElement
from shapely.geometry import Polygon, MultiPolygon
from sklearn.neighbors import KDTree
from os.path import exists, join, isfile
from os import makedirs
import pickle

from pointcloud.utils import processing, readers
from pointcloud.utils.eulerangles import euler2mat


def create_train_test_split(names, train=0.8, seed=800815):
    """
    Create train test split trough names
    :param train:
    :param names:
    :param seed:
    :return:
    """
    train_num = int(np.ceil(len(names) * train))
    test_num = int(np.ceil(len(names) * (1 - train)))
    if train_num + test_num != len(names):
        diff = len(names) - (test_num + train_num)
        test_num = test_num + diff

    random.seed(seed)
    random.shuffle(names)
    return names[0:train_num], names[train_num:train_num + test_num]


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
                                        file_format='las', file_format_settings=None, multiproc=False):
    """
    :param multiproc:
    :param file_format_settings:
    :param file_format:
    :param workspace: Path to workspace
    :param settings: Dictionary keys: step, x_pos, y_pos
    :param polygon_from_filename_settings:
    :return:
    """
    if file_format == 'las' or file_format == 'laz':
        reader = readers.LasReader(settings=file_format_settings, extension=file_format)
    elif file_format == 'txt':
        reader = readers.TxtReader(settings=file_format_settings)
    elif file_format == 'npy' or file_format == 'npz':
        reader = readers.NpyReader(settings=file_format_settings, extension=file_format)
    else:
        raise Exception('Not supported file format')
    files = glob.glob(workspace + "/*." + reader.get_extension())

    if len(files) == 0:
        raise UserWarning('No files in current workspace')

    t0 = time.time()

    if multiproc:
        pool = mp.Pool(mp.cpu_count())
        out_async = [pool.apply_async(get_polygons,
                                      args=(file, polygon_from_filename_settings, settings, reader, workspace))
                     for file in files]
        out = []
        for i, o in enumerate(out_async):
            out.append(o.get())

            dt = time.time() - t0
            avg_time_per_tile = dt / (i + 1)
            eta = avg_time_per_tile * (len(files) - i)
            sys.stdout.write(
                '\rProcessing ({:}/{:}) - total time: {:.2f} min - awg_time_per_tile: {:.2f} min; ETA: {:.2f} min\n'.
                format(i + 1, len(files), dt / 60, avg_time_per_tile / 60, eta / 60))
            sys.stdout.flush()
        pool.close()
    else:
        out = [get_polygons(file, polygon_from_filename_settings, settings, reader, workspace) for file in files]

    return out


def get_polygons(file, polygon_from_filename_settings, settings, reader, workspace):
    """

    :param file:
    :param polygon_from_filename_settings:
    :param settings:
    :param reader:
    :param workspace:
    :return:
    """
    filename = file.split('/')[-1]
    filename = filename.split('.')[0]
    polygon_file = '{:}/{:}.geojson'.format(workspace, filename)

    if os.path.isfile(polygon_file):
        # print('Reading stored one')
        with open(polygon_file, 'r') as f:
            gj = geojson.load(f)
        if gj['features'][0]['geometry'] is None:
            geom = None
        else:
            geom = Polygon(gj['features'][0]['geometry']['coordinates'][0])
        return ({'name': gj['features'][0]['properties']['name'],
                 'polygon': geom})

    polygon = calculate_polygons(file, polygon_from_filename_settings, settings, reader, workspace)
    with open(polygon_file, 'w') as f:
        features = [geojson.Feature(geometry=polygon['polygon'], properties={"name": polygon['name']})]
        feature_collection = geojson.FeatureCollection(features)
        geojson.dump(feature_collection, f)
    return polygon


def calculate_polygons(file, polygon_from_filename_settings, settings, reader, workspace):
    """

    :param file:
    :param polygon_from_filename_settings:
    :param settings:
    :param reader:
    :param workspace:
    :return:
    """
    f0 = file.split('/')[-1]
    filename = f0.split('.')[0]
    extension = f0.split('.')[1]

    if polygon_from_filename_settings is not None:
        step, x_pos, y_pos = get_polygon_from_file_settings(settings)
        polygon = calculate_polygon_from_filename(filename, step, x_pos, y_pos)
    else:
        points = reader.get_points(workspace + filename + '.' + extension)
        if np.shape(points)[0] < 3:
            print('Skipping (not enough points)')
            polygon = None
        else:
            polygon = processing.boundary(points)
    return {'name': filename, 'polygon': polygon}


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
    min_x = math.ceil(bbox[0])
    min_y = math.ceil(bbox[1])
    dx = math.ceil(bbox[2] - min_x)
    dy = math.ceil(bbox[3] - min_y)
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


def plot_3d(points, max_points=1000, title=None, save=False, path=None, labels=None, features=None,
            cmap='plasma', label_names=None, dpi=200, figsize=None, interactive=False):
    """
    Plot some nice point cloud render

    :param interactive:
    :param figsize:
    :param dpi:
    :param cmap:
    :param label_names:
    :param points:
    :param max_points:
    :param title:
    :param save:
    :param path:
    :param labels:
    :param label_vector:
    :param features:
    :param colermap:
    :return:
    """
    fig = plt.figure(dpi=dpi)
    if interactive:
        if figsize:
            w, h = figsize
        else:
            w, h = plt.rcParams['figure.figsize']
        fig.canvas.layout.height = str(h) + 'in'
        fig.canvas.layout.width = str(w) + 'in'

    ax = fig.add_subplot(111, projection='3d')
    points, labels, features = processing.sample_to_target_size(points, max_points, labels=labels, features=features)

    if label_names is not None:
        for key, l in label_names.items():
            group = points[labels == int(key)]
            color = l['color']
            label = l['name']
            if len(group) > 0:
                ax.scatter(group[:, 0], group[:, 1], group[:, 2], c=color, label=label)

    elif features is not None:
        ax.scatter([points[:, 0]], [points[:, 1]], [points[:, 2]], c=features / np.max(features), cmap=cmap)
    else:
        ax.scatter([points[:, 0]], [points[:, 1]], [points[:, 2]], c=points[:, 2] / np.max(points[:, 2]), cmap=cmap)

    if title:
        ax.set_title(title)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    if labels is not None:
        plt.legend(loc='upper left')
    # if not interactive:
    #     plt.axis('equal')

    if save:
        if path is None:
            raise Exception('Path to location has to be provided')
        plt.savefig('{0}plt_{1}.png'.format(path, title))
    else:
        plt.show()


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


def project_save_polygons(project):
    """

    :param project: project
    :return:
    """
    for pc_name, pointcloud in project.get_pointclouds().items():
        features = []

        for tile_name, tile in pointcloud.get_tiles().items():
            features.append(geojson.Feature(geometry=tile.get_polygon(), properties={"tile_name": tile_name}))

        feature_collection = geojson.FeatureCollection(features)
        with open("{:}/{:}_proj.geojson".format(pointcloud.get_workspace(), pc_name), 'w') as f:
            geojson.dump(feature_collection, f)


def _reclassify_cloud_and_create_new_cloud(tile, mappings, target_cloud):
    """

    :param tile:
    :return:
    """
    points, labels, features = tile.get_all()
    #### REMOVE THIS ####
    # TODO
    remove = [labels != 0]
    points = points[tuple(remove)]
    labels = labels[tuple(remove)]
    features = features[tuple(remove)]
    ############################

    new_labels = processing.remap_labels(labels, mappings)

    if np.shape(points)[0] < 3:
        return None

    target_cloud.create_new_tile(tile.get_name(), points=points, labels=np.squeeze(new_labels), features=features,
                                 polygon=tile.get_polygon())

    return tile


def reclassify_cloud_and_create_new_cloud(origin_cloud, target_cloud, mappings, multi_process=True):
    """

    :type origin_cloud: PointCloud
    :type target_cloud: PointCloud
    :param origin_cloud:
    :param target_cloud:
    :param mappings:
    :return:
    """
    print('\n------------\nClassify\n')
    all_tiles = origin_cloud.get_tiles().items()
    tiles_to_process = []
    for name, tile in all_tiles:
        target_cloud.add_new_tile(name, polygon=tile.get_polygon())
        if os.path.isfile("{:}/{:}.npz".format(target_cloud.get_workspace(), name)):
            continue
        tiles_to_process.append(tile)

    if multi_process:
        pool = mp.Pool(mp.cpu_count())
        tiles = [pool.apply_async(_reclassify_cloud_and_create_new_cloud, args=(tile, mappings, target_cloud,)) for tile in
                 tiles_to_process]

        t0 = time.time()
        print('\nSkipping {:} of {:}'.format(len(all_tiles) - len(tiles), len(all_tiles)))
        print('Starting to process {:} files'.format(len(tiles)))
        for i, tile in enumerate(tiles):
            tile.get()
            dt = time.time() - t0
            avg_time_per_tile = dt / (i + 1)
            eta = avg_time_per_tile * (len(tiles) - i)
            sys.stdout.write('\rProcessing {:}/{:}; total time: {:.2f} min; awg_tile: {:.2f} min; ETA: {:.2f} min\n'.
                             format(i + 1, len(tiles), dt/60, avg_time_per_tile/60, eta/60))
            sys.stdout.flush()
        pool.close()
    else:
        [_reclassify_cloud_and_create_new_cloud(tile, mappings, target_cloud) for tile in tiles_to_process]
    print('Done')


def create_and_save_kd_tree_for_tile(tile, tree_path, leaf_size=50):
    """

    :param leaf_size:
    :param tile:
    :return:
    """

    KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(tile.get_name()))

    if os.path.isfile(KDTree_file):
        return

    points = tile.get_points()

    try:
        tree = KDTree(points, leaf_size=leaf_size)
    except Exception:
        return None

    # Save KDTree
    with open(KDTree_file, 'wb') as f:
        pickle.dump(tree, f)


def create_and_save_kd_tree_for_cloud(cloud, leaf_size=50):
    """

    :param leaf_size:
    :param cloud:
    :return:
    """
    print('\n------------\nStart generating KD Trees (This may take a while)\n')
    tree_path = join(cloud.get_workspace(), 'kd_trees_{:}'.format(leaf_size))
    if not exists(tree_path):
        makedirs(tree_path)

    pool = mp.Pool(mp.cpu_count() - 2)
    tiles = [pool.apply_async(create_and_save_kd_tree_for_tile, args=(tile, tree_path, leaf_size,)) for _, tile in cloud.get_tiles().items()]

    t0 = time.time()
    print('Starting to process {:} files'.format(len(tiles)))
    for i, tile in enumerate(tiles):
        tile.get()
        gc.collect()
        dt = time.time() - t0
        avg_time_per_tile = dt / (i + 1)
        eta = avg_time_per_tile * (len(tiles) - i)
        sys.stdout.write('\rProcessing {:}/{:}; total time: {:.2f} min; awg_tile: {:.2f} min; ETA: {:.2f} min\n'.
                         format(i + 1, len(tiles), dt/60, avg_time_per_tile/60, eta/60))
        sys.stdout.flush()
    pool.close()
    print('Done')


def compute_normals_for_tile(tile):
    """

    :param tile:
    :return:
    """
    print('Processing normals for tile {:}'.format(tile.get_name()))
    points, labels, features = tile.get_all()
    normals = processing.compute_normals_for_all_points(points)
    features = np.concatenate([features, normals], axis=1)
    tile.store(points, labels, features)
    return normals, tile.get_name()


def compute_normals_for_pointcloud(pointcloud):
    """

    :type pointcloud: PointCloud
    :param pointcloud:
    :return:
    """
    pool = mp.Pool(mp.cpu_count())

    [pool.apply_async(compute_normals_for_tile, args=(tile,)) for _, tile in
     pointcloud.get_tiles().items()]
    pool.close()
    print('Done')


def calculate_single_tile_stats(tile):
    """
    TODO this should be also exploded into per tiles functions
    :return:
    """
    print('Calculating stats for tile {:}'.format(tile.get_name()))
    points, labels, features = tile.get_all()
    tile.set_number_of_points(len(points))
    point_count_per_class = processing.point_count_per_class(labels.astype(int))
    return round(tile.get_area(), 2), tile.get_number_of_points(), point_count_per_class


def calulcate_stats(pointcloud):
    """
    :type pointcloud: PointCloud
    :return:
    """
    tiles = pointcloud.get_tiles()
    stats = {}
    fq = {}
    pool = mp.Pool(mp.cpu_count())
    tile_stats_async = [pool.apply_async(calculate_single_tile_stats, args=(tile,)) for n, tile in tiles.items()]
    pool.close()
    stats['area'] = 0
    stats['num_points'] = 0
    for tile_stat_async in tile_stats_async:
        tile_stat = tile_stat_async.get()
        stats['area'] += tile_stat[0]
        stats['num_points'] += tile_stat[1]
        f = tile_stat[2]
        for c, count in f.items():
            if c in fq:
                fq[int(c)] += count
            else:
                fq[int(c)] = count
    stats['tiles'] = len(tiles)
    stats['density'] = round(stats['num_points'] / (stats['area'] + 1e-9), 2)
    stats['class_frequency'] = fq
    print('Done')
    return stats
