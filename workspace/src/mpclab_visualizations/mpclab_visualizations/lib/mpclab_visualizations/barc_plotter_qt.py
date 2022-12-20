#!/usr/bin/env python3

import numpy as np
import scipy as sp

import warnings
import pdb
from collections import deque
import itertools
from typing import List, Dict
import os

import multiprocessing as mp
import threading

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from mpclab_common.track import get_track

from mpclab_visualizations.vis_types import GlobalPlotConfigs, VehiclePlotConfigs, ObstaclePlotConfigs

class BarcFigure():
    def __init__(self,
                    t0: float = None,
                    params: GlobalPlotConfigs = GlobalPlotConfigs(),
                    logger = None) -> None:

        self.track = get_track(params.track_name)
        self.circuit = params.circuit
        self.show_meter_markers = params.show_meter_markers

        self.figure_size = params.figure_size
        self.figure_title = params.figure_title
        
        self.state_data_fields = params.state_data_fields
        self.state_units = params.state_units

        self.input_data_fields = params.input_data_fields
        self.input_units = params.input_units

        self.buffer_length = params.buffer_length
        self.keep_history = params.keep_history
        self.show_lineplots = False if (not self.state_data_fields and not self.input_data_fields) else params.show_lineplots

        self.draw_period = params.draw_period

        self.logger = logger
        self.t0 = t0

        self.dash_lib = {'solid': None, 'dash': [4,2], 'dot': [2,2], 'dash-dot': [4,2,2,2]}
        self.color_lib = {'r': (255, 0, 0), 'g': (0, 255, 0), 'b': (0, 0, 255)}

        self.vehicle_params = dict()

        self.state_trace_styles = dict()
        self.input_trace_styles = dict()

        self.vehicle_vertices = dict()
        self.vehicle_pred = dict()
        self.vehicle_full_traj = dict()
        self.vehicle_point_set = dict()
        self.vehicle_ts_data = dict()
        self.vehicle_covariance = dict()
        self.vehicle_pred_covariances = dict()

        self.running = threading.Event() # Signal for keeping I/O thread alive
        self.running.set()
        self.io_thread_init = threading.Event()

        self.plot_queue = dict() # Dict of queues for transferring data between I/O thread and plotter process
        self.new_data = dict() # Events for signaling availability of new data

        self.obstacle_params = dict()
        self.obstacle_shapes = dict()

        self.n_cov_pts = 20
        theta = np.linspace(0, 2*np.pi, self.n_cov_pts)
        self.unit_circle_pts = np.vstack((np.cos(theta), np.sin(theta)))
        self.n_std = 2

        # Mapping from number of upper triangular elements to length of diagonal
        # Used to determin the state dimension given the upper triangular of covariance matrix
        self.triu_len_to_diag_dim = {3: 2, 6: 3, 10: 4, 15: 5, 21: 6}

    # I/O thread
    def send_data_to_queue(self) -> None:
        while self.running.is_set():
            for n in self.vehicle_params.keys():
                if self.new_data[n].is_set(): # Only update if new data is available
                    data = {'rect_vertices': self.vehicle_vertices[n], 'pred': None, 'pred_cov': None, 'point_set': None, 'ts': None, 'cov': None}
                    if self.vehicle_params[n].show_pred:
                        data['pred'] = self.vehicle_pred[n]
                        if self.vehicle_params[n].show_cov:
                            data['pred_cov'] = self.vehicle_pred_covariances[n]

                    if self.vehicle_params[n].show_point_set:
                        data['point_set'] = self.vehicle_point_set[n]
                    if self.show_lineplots and self.vehicle_params[n].show_traces:
                        data['ts'] = self.vehicle_ts_data[n]
                    if self.vehicle_params[n].show_cov:
                        data['cov'] = self.vehicle_covariance[n]

                    self.plot_queue[n].put(data) # Put data into queue for plotting in separate process
                    self.new_data[n].clear() # Clear new data signal

    # This method should be run in a separate process
    def run_plotter(self) -> None:
        app = QtWidgets.QApplication([])
        pg.setConfigOptions(antialias=True, background='w')

        # figsize = (1500, 750) if self.show_lineplots else (750, 750)
        figsize = self.figure_size
        widget = pg.GraphicsLayoutWidget(show=True)
        widget.setWindowTitle(self.figure_title)
        # widget.resize(*figsize)
        widget.showMaximized()

        # Set up top view x-y plot
        l_xy = widget.addLayout()
        vb_xy = l_xy.addViewBox(lockAspect=True)
        p_xy = l_xy.addPlot(viewBox=vb_xy)
        p_xy.setDownsampling(mode='peak')
        p_xy.setLabel('left', 'Y', units='m')
        p_xy.setLabel('bottom', 'X', units='m')
        track_bbox = self.track.plot_map_qt(p_xy, close_loop=self.circuit, show_meter_markers=self.show_meter_markers)  # plot the track once at the start and avoid deleting it.
        p_xy.enableAutoRange('xy', False)
        fillLevel = track_bbox[3] + track_bbox[1]

        # Set up time history plots for states and inputs
        if self.show_lineplots:
            l_lines = widget.addLayout()
            p_lines = []
            c_lines = []
            subplots = {'state': [], 'input': []}
            for (d, u) in zip(self.state_data_fields, self.state_units):
                p = l_lines.addPlot()
                p.setDownsampling(mode='peak')
                subplots['state'].append(timeHistorySubplot(p, d, units=u, t0=self.t0, keep_history=self.keep_history))
                # cursor = pg.InfiniteLine(angle=90, movable=False)
                # p.addItem(cursor, ignoreBounds=True)
                # c_lines.append(cursor)
                p_lines.append(p)
                if len(p_lines) > 1:
                    p.setXLink(p_lines[0])
                l_lines.nextRow()
            for (d, u) in zip(self.input_data_fields, self.input_units):
                p = l_lines.addPlot()
                p.setDownsampling(mode='peak')
                subplots['input'].append(timeHistorySubplot(p, d, units=u, t0=self.t0, keep_history=self.keep_history))
                # cursor = pg.InfiniteLine(angle=90, movable=False)
                # p.addItem(cursor, ignoreBounds=True)
                # c_lines.append(cursor)
                p_lines.append(p)
                if len(p_lines) > 1:
                    p.setXLink(p_lines[0])
                l_lines.nextRow()

            if self.keep_history:
                def drag_redraw():
                    if p_lines[0].scene().lastDrag is None: return # Very hacky way to detect click-and-drag events
                    t_int = p_lines[0].getViewBox().viewRange()[0]
                    for s in subplots['state']:
                        s.drag_redraw(t_int)
                    for s in subplots['input']:
                        s.drag_redraw(t_int)
                p_lines[0].scene().sigMouseHover.connect(drag_redraw)

                def resume():
                    for s in subplots['state']:
                        s.resume()
                    for s in subplots['input']:
                        s.resume()
                p_lines[0].scene().sigMouseClicked.connect(resume)

        # Set up data cursor
        # def mouse_moved(evt):
        #     pos = evt[0]
        #     for p in p_lines:
        #         if p.sceneBoundingRect().contains(pos):
        #             mouse_point = p.vb.mapSceneToView(pos)
        #             for c in c_lines:
        #                 c.setPos(mouse_point.x())
        #             break
        # proxy = pg.SignalProxy(p_lines[0].scene().sigMouseMoved, rateLimit=60, slot=mouse_moved)

        vehicle_rects = dict()
        vehicle_predicted_trajectories = dict()
        vehicle_predicted_psi = dict()
        vehicle_full_trajectories = dict()
        vehicle_predicted_covariances = dict()
        vehicle_point_set = dict()
        vehicle_covariance = dict()
        for n in self.vehicle_params.keys():
            L = self.vehicle_params[n].vehicle_draw_L
            W = self.vehicle_params[n].vehicle_draw_W
            init_V = self.get_rect_verts(0, 0, 0, L, W)
            color_rgb = self.color_lib[self.vehicle_params[n].color]

            # Plot raceline if a file was given
            if self.vehicle_params[n].raceline_file is not None:
                raceline = np.load(os.path.expanduser(self.vehicle_params[n].raceline_file), allow_pickle=True)
                p_xy.plot(raceline['x'], raceline['y'], pen=pg.mkPen(self.vehicle_params[n].color, width=2))

            rects = dict()
            for i, t in enumerate(self.vehicle_params[n].state_topics):
                rects[t] = p_xy.plot(init_V[:,0], init_V[:,1],
                                         connect='all',
                                         pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                      dash=self.dash_lib[self.vehicle_params[n].state_trace_styles[i]]),
                                         fillLevel=fillLevel,
                                         fillBrush=color_rgb + (50,))
            vehicle_rects[n] = rects

            if self.show_lineplots and self.vehicle_params[n].show_traces:
                if self.vehicle_params[n].show_state:
                    for s in subplots['state']:
                        s.add_plot(self.buffer_length, n,
                                   self.state_trace_styles[n],
                                   self.vehicle_params[n].color)
                if self.vehicle_params[n].show_input:
                    for s in subplots['input']:
                        s.add_plot(self.buffer_length, n,
                                   self.input_trace_styles[n],
                                   self.vehicle_params[n].color)
            
            preds = dict()
            if self.vehicle_params[n].show_pred:
                for i, t in enumerate(self.vehicle_params[n].pred_topics):
                    preds[t] = p_xy.plot([], [], pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                                dash=self.dash_lib[self.vehicle_params[n].pred_styles[i]]),
                                        symbolBrush=self.vehicle_params[n].color, symbolPen='k',
                                        symbol='o', symbolSize=5)
                    vehicle_predicted_psi[n] = p_xy.plot([], [],
                                                        connect='all',
                                                        pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                                    dash=self.dash_lib['solid']),
                                                        symbolBrush=None, symbolPen=None)
                    vehicle_predicted_covariances[n] = [p_xy.plot([], [],
                                                                pen=pg.mkPen(self.vehicle_params[n].color, width=1e-3,
                                                                            dash=self.dash_lib['solid']),
                                                                fillLevel=fillLevel, fillBrush=color_rgb + (50,)) for _ in
                                                        range(self.n_cov_pts)]
                vehicle_predicted_trajectories[n] = preds
            
            point_sets = dict()
            if self.vehicle_params[n].show_point_set:
                for t, m in zip(self.vehicle_params[n].point_set_topics, self.vehicle_params[n].point_set_modes):
                    if m == 'points':
                        point_sets[t] = p_xy.plot([], [], pen=None, symbolBrush=None,
                                                        symbolPen=self.vehicle_params[n].color, symbol='s', symbolSize=6)
                    elif m == 'hull':
                        point_sets[t] = p_xy.plot([], [],
                                            connect='all',
                                            pen=pg.mkPen(self.vehicle_params[n].color, width=1,
                                                        dash=self.dash_lib[self.vehicle_params[n].state_trace_styles[i]]),
                                            fillLevel=fillLevel,
                                            fillBrush=color_rgb + (50,))
                    else:
                        raise(ValueError("Point set mode must be 'points' or 'hull'"))
                vehicle_point_set[n] = point_sets

            # Add an empty line plot for each covariance ellipse
            covs = dict()
            if self.vehicle_params[n].show_cov:
                for t in self.vehicle_params[n].cov_topics:
                    covs[t] = p_xy.plot([], [], 
                                        pen=pg.mkPen(self.vehicle_params[n].color, width=1, dash=self.dash_lib['solid']),
                                        symbolBrush=None, symbolPen=None)
                vehicle_covariance[n] = covs

        # redraw method is attached to timer with period self.draw_period
        def redraw():
            for n in self.vehicle_params.keys():
                if not self.plot_queue[n].empty(): # Don't redraw if data queue is empty
                    while not self.plot_queue[n].empty(): # Read queue until empty
                        data = self.plot_queue[n].get()
                    
                    # Draw vehicle rectangles
                    rect_vertices = data['rect_vertices']
                    if rect_vertices is not None:
                        for t in self.vehicle_params[n].state_topics:
                            if t in rect_vertices.keys():
                                vehicle_rects[n][t].setData(rect_vertices[t][:,0], rect_vertices[t][:,1])

                    # Draw state and input traces
                    ts_data = data['ts']
                    if ts_data is not None:
                        if ts_data['state']:
                            for s in subplots['state']:
                                s.update(n, ts_data['state'])
                                s.redraw()
                        if ts_data['input']:
                            for s in subplots['input']:
                                s.update(n, ts_data['input'])
                                s.redraw()

                    # Draw predictions
                    pred = data['pred']
                    if pred is not None:
                        for t in self.vehicle_params[n].pred_topics:
                            vehicle_predicted_trajectories[n][t].setData(pred[t]['x'], pred[t]['y'])
                            if self.show_lineplots:
                                for s in subplots['state']:
                                    s.update_seq(n, pred[t])
                                    s.redraw()
                                for s in subplots['input']:
                                    s.update_seq(n, pred[t])
                                    s.redraw()
                            if self.vehicle_params[n].show_full_vehicle_bodies:
                                x_s = []; y_s = []
                                for i in range(len(pred[t]['x'])):
                                    a = self.get_rect_verts(pred[t]['x'][i], pred[t]['y'][i], pred[t]['psi'][i], L, W)
                                    x_s.extend(a[:, 0])
                                    y_s.extend(a[:, 1])
                                vehicle_predicted_psi[n].setData(x_s, y_s)

                            pred_cov = data['pred_cov']
                            if pred_cov is not None:
                                for (i, pc) in enumerate(pred_cov):
                                    vehicle_predicted_covariances[n][i].setData(pc['x'], pc['y'])
                    
                    if self.vehicle_params[n].show_full_traj:
                        vehicle_full_trajectories[n].setData(self.vehicle_full_traj[n]['x'],self.vehicle_full_traj[n]['y'])
                    
                    # Draw point set
                    point_set = data['point_set']
                    if point_set is not None:
                        for t in self.vehicle_params[n].point_set_topics:
                            vehicle_point_set[n][t].setData(point_set[t]['x'], point_set[t]['y'])

                    # Draw covariance ellipse
                    cov = data['cov']
                    if cov is not None:
                        for t in self.vehicle_params[n].cov_topics:
                            vehicle_covariance[n][t].setData(cov[t]['x'], cov[t]['y'])

        timer = QtCore.QTimer()
        timer.timeout.connect(redraw)
        timer.start(self.draw_period*1000)

        app.exec_()

        return

    def add_vehicle(self,
                params: VehiclePlotConfigs = VehiclePlotConfigs()) -> str:

        L = params.vehicle_draw_L
        W = params.vehicle_draw_W
        name = params.name

        self.vehicle_params[name] = params

        init_V = self.get_rect_verts(0, 0, 0, L, W)
        rect_vertices = dict()
        for i, t in enumerate(params.state_topics):
            rect_vertices[t] = init_V
        self.vehicle_vertices[name] = rect_vertices

        if params.show_traces:
            if params.show_state:
                state_trace_styles= dict()
                for i, t in enumerate(params.state_topics):
                    state_trace_styles[t] = params.state_trace_styles[i]
                self.state_trace_styles[name] = state_trace_styles
        
            if params.show_input:
                input_trace_styles = dict()
                for i, t in enumerate(params.input_topics):
                    input_trace_styles[t] = params.input_trace_styles[i]
                self.input_trace_styles[name] = input_trace_styles

        if params.show_cov:
            self.vehicle_covariance[name] = {t: {'x': [], 'y': []} for t in params.cov_topics}

        if params.show_full_traj:
            self.vehicle_full_traj[name] = {'x': [], 'y': [], 'psi': []}

        if params.show_pred:
            self.vehicle_pred[name] = {t: {'x': [], 'y': [], 'psi': []} for t in params.pred_topics}
            if params.show_cov:
                self.vehicle_pred_covariances[name] = {t: [{'x': [], 'y': []}] for t in params.pred_topics}

        if params.show_point_set:
            self.vehicle_point_set[name] = {t: {'x': [], 'y': [], 'psi': []} for t in params.point_set_topics}

        self.plot_queue[name] = mp.Queue() # Create queue for vehicle plot data
        self.new_data[name] = threading.Event() # Create event to signal new data for vehicle

        return name

    def update_vehicle_state(self, vehicle_name: str, vehicle_data: dict) -> None:
        if vehicle_name in self.vehicle_params.keys():
            vehicle_params = self.vehicle_params[vehicle_name]

            ts_data = {'state': dict(), 'input': dict()}
            vehicle_vertices = dict()
            xy_cov_ell_pts = np.empty((0, 2))

            L = vehicle_params.vehicle_draw_L
            W = vehicle_params.vehicle_draw_W

            for t in vehicle_params.state_topics:
                if t in vehicle_data.keys():
                    topic_data = vehicle_data[t]
                    if topic_data.t is not None:
                        if None in (topic_data.x.x, topic_data.x.y, topic_data.e.psi):
                            topic_data.x.x, topic_data.x.y, topic_data.e.psi = self.track.local_to_global([topic_data.p.s, topic_data.p.x_tran, topic_data.p.e_psi])
                        vehicle_vertices[t] = self.get_rect_verts(topic_data.x.x, topic_data.x.y, topic_data.e.psi, L, W)
                        if vehicle_params.show_state:
                            ts_data['state'][t] = topic_data

            if vehicle_params.show_input:
                for t in vehicle_params.input_topics:
                    if t in vehicle_data.keys():
                        topic_data = vehicle_data[t]
                        if topic_data.t is not None:
                            ts_data['input'][t] = topic_data

            self.vehicle_vertices[vehicle_name] = vehicle_vertices
            self.vehicle_ts_data[vehicle_name] = ts_data

            if vehicle_params.show_cov:
                for t in vehicle_params.cov_topics:
                    topic_data = vehicle_data[t]
                    if topic_data.global_state_covariance is not None and topic_data.local_state_covariance is not None:
                        raise RuntimeError('State covariance matrix provided in both local and global frames')
                    if topic_data.global_state_covariance is not None:
                        cov_triu = np.array(topic_data.global_state_covariance)
                        n = self.triu_len_to_diag_dim[len(cov_triu)]
                        cov = np.zeros((n, n))
                        cov[np.triu_indices(n)] = cov_triu
                        xy_cov = cov[:2,:2]
                        xy_cov_ell_pts = np.array([topic_data.x.x, topic_data.x.y]) + self.n_std*np.linalg.cholesky(xy_cov).dot(self.unit_circle_pts).T
                    elif topic_data.local_state_covariance is not None:
                        cov_triu = np.array(topic_data.local_state_covariance)
                        n = self.triu_len_to_diag_dim[len(cov_triu)]
                        cov = np.zeros((n, n))
                        cov[np.triu_indices(n)] = cov_triu
                        sey_cov = cov[-2:,-2:]
                        sey_cov_ell_pts = np.array([topic_data.p.s, topic_data.p.x_tran]) + self.n_std*np.linalg.cholesky(sey_cov).dot(self.unit_circle_pts).T
                        xy_cov_ell_pts = np.array([self.track.local_to_global(np.append(sey_cov_ell_pts[i], 0))[:2] for i in range(self.n_cov_pts)])

                    self.vehicle_covariance[vehicle_name][t]['x'] = xy_cov_ell_pts[:,0]
                    self.vehicle_covariance[vehicle_name][t]['y'] = xy_cov_ell_pts[:,1]
        else:
            warnings.warn("Vehicle name '%s' not recognized, nothing to update..." % vehicle_name, UserWarning)

    def update_vehicle_prediction(self, vehicle_name: str, vehicle_data: dict) -> None:
        for t in self.vehicle_params[vehicle_name].pred_topics:
            if t in vehicle_data.keys():
                pred_data = vehicle_data[t]
                if pred_data.t is not None:
                    if pred_data.x is None or pred_data.y is None or len(pred_data.x) == 0 or len(pred_data.y) == 0: # x, y, or psi fields are empty
                        x_pred, y_pred, psi_pred = [], [], []
                        for i in range(len(pred_data.s)):
                            (x, y, psi) = self.track.local_to_global([pred_data.s[i], pred_data.x_tran[i], pred_data.e_psi[i]])
                            x_pred.append(x); y_pred.append(y); psi_pred.append(psi)
                    else:
                        x_pred, y_pred, psi_pred = pred_data.x, pred_data.y, pred_data.psi
                    self.vehicle_pred[vehicle_name][t]['x'], self.vehicle_pred[vehicle_name][t]['y'], self.vehicle_pred[vehicle_name][t]['psi'] = x_pred, y_pred, psi_pred

                    self.vehicle_pred[vehicle_name][t]['t'] = pred_data.t
                    self.vehicle_pred[vehicle_name][t]['s'] = pred_data.s
                    self.vehicle_pred[vehicle_name][t]['x_tran'] = pred_data.x_tran
                    self.vehicle_pred[vehicle_name][t]['e_psi'] = pred_data.e_psi
                    self.vehicle_pred[vehicle_name][t]['w_psi'] = pred_data.psidot
                    self.vehicle_pred[vehicle_name][t]['v_long'] = pred_data.v_long
                    self.vehicle_pred[vehicle_name][t]['v_tran'] = pred_data.v_tran
                    self.vehicle_pred[vehicle_name][t]['u_a'] = pred_data.u_a
                    self.vehicle_pred[vehicle_name][t]['u_steer'] = pred_data.u_steer

        # Update covariance ellipses over prediction horizon
        if self.vehicle_params[vehicle_name].show_cov:
            if pred_data.global_state_covariance is not None and pred_data.local_state_covariance is not None:
                raise RuntimeError('State covariance matrix provided in both local and global frames')
            N = len(x_pred)
            ell_pts = [None for _ in range(N-1)]
            if pred_data.global_state_covariance is not None:
                cov_triu_seq = np.array(pred_data.global_state_covariance).reshape((N,-1))
                n = self.triu_len_to_diag_dim[cov_triu_seq.shape[1]]
                cov = np.zeros((n, n))
                for i in range(1, N):
                    cov[np.triu_indices(n)] = cov_triu_seq[i]
                    xy_cov = cov[:2,:2] + cov[:2,:2].T - np.diag(np.diag(cov)[:2])
                    xy_cov_ell_pts = np.array([pred_data.x[i], pred_data.y[i]]) + self.n_std*np.linalg.cholesky(xy_cov).dot(self.unit_circle_pts).T
                    ell_pts[i-1] = {'x': xy_cov_ell_pts[:,0], 'y': xy_cov_ell_pts[:,1]}
            elif pred_data.local_state_covariance is not None:
                cov_triu_seq = np.array(pred_data.local_state_covariance).reshape((N,-1))
                n = self.triu_len_to_diag_dim[cov_triu_seq.shape[1]]
                cov = np.zeros((n, n))
                for i in range(1, N):
                    cov[np.triu_indices(n)] = cov_triu_seq[i]
                    sey_cov = cov[-2:,-2:] + cov[-2:,-2:].T - np.diag(np.diag(cov)[-2:])
                    sey_cov_ell_pts = np.array([pred_data.s[i], pred_data.x_tran[i]]) + self.n_std*np.linalg.cholesky(sey_cov).dot(self.unit_circle_pts).T
                    xy_cov_ell_pts = np.array(
                        [self.track.local_to_global([sey_cov_ell_pts[j][0], sey_cov_ell_pts[j][1], 0])[:2] for j in
                         range(self.n_cov_pts)])
                    ell_pts[i-1] = {'x': xy_cov_ell_pts[:,0], 'y': xy_cov_ell_pts[:,1]}
            elif pred_data.xy_cov is not None:
                xy_unflat = np.array(pred_data.xy_cov).reshape(N, 4)
                for i in range(1, N):
                    xy_cov = xy_unflat[i].reshape(2, 2)
                    angle = psi_pred[i] # + pred_data.e_psi[i]
                    rot_mat = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
                    cov = xy_cov
                    # print("bodycov", cov)
                    cov_ell_pts = (self.n_std * np.sqrt(
                        cov).dot(self.unit_circle_pts).T).transpose()

                    cov_ell_pts = (rot_mat@cov_ell_pts).transpose()
                    xy_cov_ell_pts = np.array([x_pred[i], y_pred[i]]) + cov_ell_pts

                    ell_pts[i - 1] = {'x': xy_cov_ell_pts[:, 0], 'y': xy_cov_ell_pts[:, 1]}
            # elif pred_data.sey_cov is not None:
            #     sey_unflat = np.array(pred_data.sey_cov).reshape(N, 4)
            #     for i in range(1, N):
            #         sey_cov = sey_unflat[i].reshape(2, 2)
            #         # print(sey_cov)
            #         sey_cov_ell_pts = np.array([pred_data.s[i], pred_data.x_tran[i]]) + self.n_std * np.linalg.cholesky(
            #             sey_cov).dot(self.unit_circle_pts).T
            #         xy_cov_ell_pts = np.array([self.track.local_to_global([sey_cov_ell_pts[j][0],sey_cov_ell_pts[j][1], 0])[:2] for j in
            #                                    range(self.n_cov_pts)])
            #         ell_pts[i - 1] = {'x': xy_cov_ell_pts[:, 0], 'y': xy_cov_ell_pts[:, 1]}

            self.vehicle_pred_covariances[vehicle_name] = ell_pts

    def update_vehicle_point_sets(self, vehicle_name: str, vehicle_data: dict) -> None:
        for t, m in zip(self.vehicle_params[vehicle_name].point_set_topics, self.vehicle_params[vehicle_name].point_set_modes):
            point_set_data = vehicle_data[t]
            if point_set_data.t is not None:
                if not point_set_data.x or not point_set_data.y or not point_set_data.psi:
                    hull = sp.spatial.ConvexHull(np.array([point_set_data.s, point_set_data.x_tran]).T)
                    s_ss, ey_ss, ep_ss = np.array(point_set_data.s)[hull.vertices], np.array(point_set_data.x_tran)[hull.vertices], np.array(point_set_data.e_psi)[hull.vertices]
                    x_ss, y_ss, psi_ss = [], [], []
                    for i in range(len(s_ss)):
                        (x, y, psi) = self.track.local_to_global([s_ss[i], ey_ss[i], ep_ss[i]])
                        x_ss.append(x); y_ss.append(y); psi_ss.append(psi)
                else:
                    hull = sp.spatial.ConvexHull(np.array([point_set_data.x, point_set_data.y]).T)
                    x_ss, y_ss, psi_ss = np.array(point_set_data.x)[hull.vertices], np.array(point_set_data.y)[hull.vertices], np.array(point_set_data.psi)[hull.vertices]
                if m == 'hull':
                    x_ss = np.append(x_ss, x_ss[0])
                    y_ss = np.append(y_ss, y_ss[0])
                    psi_ss = np.append(psi_ss, psi_ss[0])
                self.vehicle_point_set[vehicle_name][t]['x'], self.vehicle_point_set[vehicle_name][t]['y'], self.vehicle_point_set[vehicle_name][t]['psi'] = x_ss, y_ss, psi_ss

    def update(self, vehicle_name: str, vehicle_data: dict) -> None:
        if vehicle_name in self.vehicle_params.keys():
            self.update_vehicle_state(vehicle_name, vehicle_data)
            if self.vehicle_params[vehicle_name].show_pred:
                self.update_vehicle_prediction(vehicle_name, vehicle_data)
            if self.vehicle_params[vehicle_name].show_point_set:
                self.update_vehicle_point_sets(vehicle_name, vehicle_data)
            self.new_data[vehicle_name].set() # Signal update thread that new data is ready
        else:
            warnings.warn("Vehicle name '%s' not recognized, nothing to update..." % vehicle_name, UserWarning)

        # Start I/O thread on first call of update
        if not self.io_thread_init.is_set():
            self.update_thread = threading.Thread(target=self.send_data_to_queue)
            self.update_thread.start()
            self.io_thread_init.set()

    def get_rect_verts(self, x_center: float, y_center: float, theta: float, L: float, W: float) -> np.ndarray:
        if None in (x_center, y_center, theta, L, W):
            return None

        tr = [x_center + L*np.cos(theta)/2 - W*np.sin(theta)/2, y_center + L*np.sin(theta)/2 + W*np.cos(theta)/2]
        br = [x_center + L*np.cos(theta)/2 + W*np.sin(theta)/2, y_center + L*np.sin(theta)/2 - W*np.cos(theta)/2]
        tl = [x_center - L*np.cos(theta)/2 - W*np.sin(theta)/2, y_center - L*np.sin(theta)/2 + W*np.cos(theta)/2]
        bl = [x_center - L*np.cos(theta)/2 + W*np.sin(theta)/2, y_center - L*np.sin(theta)/2 - W*np.cos(theta)/2]

        return np.array([bl, br, tr, tl, bl])

    def run(self):
        # Create plotter and start Qt event loop in separate process
        p = mp.Process(target=self.run_plotter)
        p.start()

class timeHistorySubplot():
    '''
    creates a scrollingPlot() objects and adds data to it from structures
    adds the variabe 'var' from data objects passed to it

    ex: if var = 'x', it will plot sim_data.x, est_data.x, and mea_data.x
    '''
    def __init__(self, p: pg.PlotDataItem, var: str, 
                    units: str = None, 
                    keep_history: bool = False,
                    t0: float = None):
        self.p = p
        self.plotters = dict()
        self.plotter_trace_styles = dict()
        self.var = var
        self.t0 = t0
        self.keep_history = keep_history

        self.p.setLabel('left', var, units=units)

    '''
    Adds a scrollingPlot to the subplot. The plot may contain multiple traces, whose
    names are specified in the list trace_names
    '''
    def add_plot(self, n_pts: int,
                plot_name: str = 'plot_1',
                trace_styles: Dict[str,str] = {'trace_1': 'solid'},
                trace_color: str = 'b') -> None:
        if plot_name not in self.plotters.keys():
            plotter = scrollingPlot(self.p, n_pts, trace_styles, trace_color, t0=self.t0, keep_history=self.keep_history)
            self.plotters[plot_name] = plotter
            self.plotter_trace_styles[plot_name] = trace_styles
        else:
            warnings.warn("Plot with the name '%s' already exists, not adding..." % plot_name, UserWarning)

    def update(self, plot_name: str, data: dict) -> None:
        # adds data to the plot, which will remove oldest data points
        # pass 'None' type to avoid updating
        if plot_name in self.plotters.keys() and data:
            for n in data.keys():
                if n in self.plotter_trace_styles[plot_name].keys():
                    if data[n] is not None:
                        if '.' in self.var:
                            field_names = self.var.split('.')
                            d = data[n]
                            for s in field_names:
                                d = getattr(d, s)
                        else:
                            d = getattr(data[n], self.var)
                        self.plotters[plot_name].add_point(data[n].t, d, trace_name=n)
                else:
                    warnings.warn("Trace name '%s' not recognized, nothing to update..." % n, UserWarning)
        else:
           warnings.warn("Plot name '%s' not recognized, nothing to update..." % plot_name, UserWarning)

    def update_seq(self, plot_name: str, data: dict) -> None:
        # adds data to the plot, which will remove oldest data points
        # pass 'None' type to avoid updating
        if plot_name in self.plotters.keys():
            if '.' in self.var:
                var = self.var.split('.')[-1]
            else:
                var = self.var
            if var in data.keys():
                self.plotters[plot_name].add_seq(data['t'], data[var])
        # else:
        #    warnings.warn("Plot name '%s' not recognized, nothing to update..." % plot_name, UserWarning)

    def redraw(self) -> None:
        for p in self.plotters.values():
            p.redraw()

    def drag_redraw(self, t_int) -> None:
        for p in self.plotters.values():
            p.drag_redraw(t_int)

    def resume(self) -> None:
        for p in self.plotters.values():
            p.resume()

class scrollingPlot():
    '''
    Creates and updates a scrolling plot on 'ax'
    num_pts: size of the list to be used for each trace
    num_traces: number of different input to plot

    Example: when comparing simulated and estimated velocity, there would be two traces
    and however many prior points are desired (perhaps 50)
    '''
    def __init__(self, p: pg.PlotDataItem, num_pts: int, trace_styles: Dict[str, str],
                    trace_color: str = 'b',
                    keep_history: bool = False,
                    t0: float = None):
        '''
        Warning: time.time() in a function argument corresponds to the time when the function was created
        (when the python interpreter started)
        It is only suitable for initializers!!!!!!! Do not use for methods used to update.
        '''

        self.p = p
        self.trace_styles = trace_styles
        self.trace_color = trace_color
        self.t0 = t0
        self.init = True
        self.dt = 0.1 # Hard coded for now
        self.num_pts = num_pts
        self.keep_history = keep_history
        self.pause_redraw = False

        dash_lib = {'solid': None, 'dash': [4,2], 'dot': [2,2], 'dash-dot': [4,2,2,2]}
        #WARNING: anything of this form will wind up with pass-by-reference issues
        #self.pt_memory   = [[None]*num_pts]*num_traces
        #self.time_memory = [[None]*num_pts]*num_traces

        self.pt_memory = dict()
        self.time_memory = dict()
        self.lines = dict()
        if keep_history:
            self.pt_history = dict()
            self.time_history = dict()

        for (n, s) in trace_styles.items():
            self.pt_memory[n] = deque([], num_pts)
            self.time_memory[n] = deque([], num_pts)
            self.lines[n] = self.p.plot(self.time_memory[n], self.pt_memory[n],
                                        pen=pg.mkPen(trace_color, width=1, dash=dash_lib[s]))

            if keep_history:
                self.pt_history[n] = []
                self.time_history[n] = []

        self.seq_memory = np.array([])
        self.seq_time_memory = np.array([])
        self.seq_line = self.p.plot(self.seq_time_memory, self.seq_memory,
                                        pen=pg.mkPen('k', width=1, dash=None),
                                        symbolBrush=trace_color, 
                                        symbolPen=trace_color,
                                        symbol='o', 
                                        symbolSize=4)

    def add_point(self, t: float, x: float, trace_name: str) -> None:
        if trace_name not in self.trace_styles.keys():
            return
        if x is None or t is None or t < 0:
            return

        if self.t0 is not None:
            t = t - self.t0

        if len(self.pt_memory[trace_name]) >= self.num_pts and self.keep_history:
            self.pt_history[trace_name].append(self.pt_memory[trace_name].popleft())
            self.time_history[trace_name].append(self.time_memory[trace_name].popleft())

        self.pt_memory[trace_name].append(x)
        self.time_memory[trace_name].append(t)

        return
    
    def add_seq(self, t: float, x: np.ndarray) -> None:
        if x is None or t is None or t < 0:
            return

        if self.t0 is not None:
            t = t - self.t0

        self.seq_memory = x
        self.seq_time_memory = t + self.dt*np.arange(len(x))

        return

    def redraw(self) -> None:
        if not self.pause_redraw:
            for n in self.trace_styles.keys():
                self.lines[n].setData(self.time_memory[n], self.pt_memory[n])
            if len(self.seq_memory) > 0:
                self.seq_line.setData(self.seq_time_memory, self.seq_memory)
    
    def drag_redraw(self, t_int):
        ts, tf = t_int
        for n in self.trace_styles.keys():
            if ts < self.time_memory[n][0] and tf < self.time_memory[n][0]:
                if len(self.time_history[n]) > 0:
                    s_idx = np.argmin(np.abs(np.array(self.time_history[n]) - ts))
                    e_idx = np.argmin(np.abs(np.array(self.time_history[n]) - tf))
                    time_data = self.time_history[n][s_idx:e_idx]
                    pt_data = self.pt_history[n][s_idx:e_idx]
                else:
                    time_data, pt_data = [], []
            elif ts < self.time_memory[n][0] and tf >= self.time_memory[n][0]:
                e_idx = np.argmin(np.abs(np.array(self.time_memory[n]) - tf))
                if len(self.time_history[n]) > 0:
                    s_idx = np.argmin(np.abs(np.array(self.time_history[n]) - ts))
                    time_data = self.time_history[n][s_idx:] + list(itertools.islice(self.time_memory[n], 0, e_idx))
                    pt_data = self.pt_history[n][s_idx:] + list(itertools.islice(self.pt_memory[n], 0, e_idx))
                else:
                    time_data = list(itertools.islice(self.time_memory[n], 0, e_idx))
                    pt_data = list(itertools.islice(self.pt_memory[n], 0, e_idx))
            else:
                s_idx = np.argmin(np.abs(np.array(self.time_memory[n]) - ts))
                e_idx = np.argmin(np.abs(np.array(self.time_memory[n]) - tf))
                time_data = list(itertools.islice(self.time_memory[n], s_idx, e_idx))
                pt_data = list(itertools.islice(self.pt_memory[n], s_idx, e_idx))
            
            self.lines[n].setData(time_data, pt_data)
        self.pause_redraw = True
    
    def resume(self):
        self.pause_redraw = False

if __name__ == '__main__':
    pass