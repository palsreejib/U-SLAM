from setuptools import setup

package_name = 'mps_sim'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='MPS simulation package',
    license='MIT',
    entry_points={
        # NOTE: this package builds with ament_cmake (see CMakeLists.txt),
        # which installs these scripts directly via install(PROGRAMS ...).
        # This entry_points list is not what actually wires up the
        # `ros2 run mps_sim <script>.py` commands — it's kept here only
        # for documentation / in case someone repoints the build to
        # ament_python later.
        'console_scripts': [
            'mps_driver_node = mps_driver.mps_driver_node:main',
            'ground_truth_node = mps_driver.ground_truth_node:main',
            'dvl_node = mps_driver.dvl_node:main',
            'pressure_node = mps_driver.pressure_node:main',
            'sync_node = mps_driver.sync_node:main',
        ],
    },
)
