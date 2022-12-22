from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory

from mpclab_common.mpclab_base_nodes import read_yaml_file

import pathlib, os
from datetime import datetime

exp_name = 'barc_sim_project'

launch_files_dir = get_package_share_directory('barc_launch')
config_dir = os.path.join(launch_files_dir, 'config', exp_name)
rosbag_dir = '/project_data/' + exp_name + datetime.now().strftime('_%m-%d-%Y_%H-%M-%S')

global_params_file = os.path.join(config_dir, 'global_params.yaml')
global_params = read_yaml_file(global_params_file)

def generate_launch_description():
    return LaunchDescription([
        # BARC 1
        Node(
            package='mpclab_simulation',
            namespace='experiment/barc_1',
            executable='py_sim_dynamics_node.py',
            name='barc_1_simulator',
            parameters=[os.path.join(config_dir,'barc_1/vehicle_simulator.yaml')]+global_params
        ),
        Node(
            package='mpclab_simulation',
            namespace='experiment/barc_1',
            executable='py_sim_optitrack_node.py',
            name='barc_1_optitrack',
            parameters=[os.path.join(config_dir,'barc_1/optitrack_simulator.yaml')]+global_params
        ),

        Node(
            package='mpclab_estimation',
            namespace='experiment/barc_1',
            executable='py_optitrack_pass_through_estimator_node.py',
            name='barc_1_estimator',
            parameters=[os.path.join(config_dir,'barc_1/estimator.yaml')]+global_params
        ),
        Node(
            package='mpclab_controllers',
            namespace='experiment/barc_1',
            executable='py_project_node.py',
            name='barc_1_control'
        ),

        # Global
        Node(
            package='mpclab_visualizations',
            namespace='experiment',
            executable='py_vis_qt_node.py',
            name='visualizer',
            parameters=[os.path.join(config_dir,'visualization.yaml')]+global_params
        ),

        ExecuteProcess(
            cmd=['ros2', 'bag', 'record',
                    '-o', rosbag_dir,
                    '--qos-profile-overrides-path', os.path.join(config_dir, 'qos_settings.yaml'),
                    '/experiment/barc_1/state_input_log',
                    '/experiment/barc_1/est_state',
                    '/experiment/barc_1/ecu',
                    '/experiment/barc_1/pred',
                    '/experiment/barc_1/ref']
        )
    ])
