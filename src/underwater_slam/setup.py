from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'underwater_slam'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'config'),
            glob(os.path.join('config', '*.rviz'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sreejib',
    maintainer_email='sreejib@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "dvl_publisher = underwater_slam.dvl_publisher:main",
            "sonar_publisher = underwater_slam.sonar_publisher:main",
            "image_publisher = underwater_slam.image_publisher:main",
            "slam = underwater_slam.slam:main"
        ],
    },
)
