from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name='pointcloud',
    version='0.1',
    description='Manage, visualize and process point cloud projects',
    long_description=long_description,
    license='MIT',
    author='Jernej Nejc Dougan',
    author_email='nejc.dougan@gmail.com',
    url='https://github.com/nejcd',
    keywords='remote sensing lidar point cloud geo spatial',
    packages=find_packages(exclude=["*tests*"]),
    install_requires=[
        'numpy',
        'shapely',
        'scipy',
        'laspy',
        'matplotlib',
        'descartes'
    ],
    scripts=[
        'scripts/extract_classes.py',
        'scripts/multipolygon.py',
    ],
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: OpenStack',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: GIS',

    ]

)