from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

with open("README.md") as f:
    long_description = f.read()

extensions = [
    Extension(
        "pointcloud.utils.processing",
        ["pointcloud/utils/processing.pyx"],
    ),
]

setup(
    name='pointcloud',
    version='0.1',
    description='Manage, visualize and process point cloud projects',
    long_description=long_description,
    license='MIT',
    author='Jernej Nejc Dougan',
    author_email='nejc.dougan@gmail.com',
    url='https://github.com/nejcd/pointcloud.git',
    keywords='remote sensing lidar point cloud geo spatial',
    packages=find_packages(exclude=["*tests*"]),
    ext_modules=cythonize(extensions),
    install_requires=[
        'numpy',
        'shapely',
        'scipy',
        'laspy',
        'matplotlib',
        'descartes',
        'plyfile',
        'geojson',
        'sklearn'
        'cython'
    ],
    scripts=[
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: OpenStack',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',
    ]
)
