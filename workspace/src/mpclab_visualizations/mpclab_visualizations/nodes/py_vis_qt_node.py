#!/usr/bin/env python3

import rclpy
from rclpy.qos import qos_profile_sensor_data

from typing import List

from mpclab_visualizations.barc_plotter_qt import BarcFigure
from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs

from mpclab_common.mpclab_base_nodes import MPClabNode
from mpclab_common.pytypes import VehicleState, VehicleActuation, VehiclePrediction
from mpclab_common.msg import VehicleStateMsg, VehicleActuationMsg, PredictionMsg

class VisNodeParams():
    def __init__(self):
        self.track_name = None
        self.global_plot_params = GlobalPlotConfigs()

    def load_names(self, vehicle_namespaces: List[str]):
        '''
        creates a template entry for every vehicle name

        intention is that vehicle_namespaces is loaded first (VehicleNameParams), and then this class loads the rest
        this function should help set up the vehicle names.
        '''
        for name in vehicle_namespaces:
            object.__setattr__(self, name, VehiclePlotConfigs())
        return

class VisNode(MPClabNode):

    def __init__(self):
        super().__init__('visualization')

        namespace = self.get_namespace()

        # Get vehicle namespaces
        self.declare_parameter(name='vehicle_namespaces')
        self.vehicle_namespaces = self.get_parameter('vehicle_namespaces').value

        # Load parameters
        param_template = VisNodeParams()
        param_template.load_names(self.vehicle_namespaces)
        self.autodeclare_parameters(param_template, namespace)
        self.autoload_parameters(param_template, namespace)
        
        self.global_plot_params.track_name = self.track_name

        # Get handle to ROS clock
        self.clock = self.get_clock()

        # Initialize
        self.barc_fig = BarcFigure(t0=self.clock.now().nanoseconds/1E9, 
                                    params=self.global_plot_params,
                                    logger=self.get_logger().info)

        self.vehicle_data = {n: dict() for n in self.vehicle_namespaces}

        # Add vehicles to plotter and create subscribers for each vehicle
        for n in self.vehicle_namespaces:
            vehicle_plot_params = self.__getattribute__(n)
            vehicle_plot_params.name = n

            self.barc_fig.add_vehicle(params=vehicle_plot_params)

            if vehicle_plot_params.show_state:
                for t in vehicle_plot_params.state_topics:
                    self.vehicle_data[n][t] = VehicleState()
                    self.create_subscription(
                        VehicleStateMsg,
                        '/'.join((namespace, n, t)),
                        self.get_callback(n, t),
                        qos_profile_sensor_data)

            if vehicle_plot_params.show_input:
                for t in vehicle_plot_params.input_topics:
                    self.vehicle_data[n][t] = VehicleActuation()
                    self.create_subscription(
                        VehicleActuationMsg,
                        '/'.join((namespace, n, t)),
                        self.get_callback(n, t),
                        qos_profile_sensor_data)

            if vehicle_plot_params.show_pred:
                for t in vehicle_plot_params.pred_topics:
                    self.vehicle_data[n][t] = VehiclePrediction()
                    self.create_subscription(
                        PredictionMsg,
                        '/'.join((namespace, n, t)),
                        self.get_callback(n, t),
                        qos_profile_sensor_data)

            if vehicle_plot_params.show_point_set:
                for t in vehicle_plot_params.point_set_topics:
                    self.vehicle_data[n][t] = VehiclePrediction()
                    self.create_subscription(
                        PredictionMsg,
                        '/'.join((namespace, n, t)),
                        self.get_callback(n, t),
                        qos_profile_sensor_data)
        self.barc_fig.run()

        # Attach update function to timer
        self.update_timer = self.create_timer(self.global_plot_params.update_period, self.update)

        self.init = True

    def get_callback(self, name, topic):
        def callback(msg):
            self.unpack_msg(msg, self.vehicle_data[name][topic])
        return callback

    def update(self):
        if self.init:
            self.get_logger().info('===== Visualizer start =====')
            self.init = False
        for name in self.vehicle_namespaces:
            self.barc_fig.update(name, self.vehicle_data[name])

        return

def main(args=None):
    rclpy.init(args=args)
    vis = VisNode()
    rclpy.spin(vis)

    vis.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
