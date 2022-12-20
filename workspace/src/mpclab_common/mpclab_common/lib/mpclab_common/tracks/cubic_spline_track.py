import pdb
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
import pyqtgraph as pg
from scipy.interpolate import CubicSpline
from scipy import optimize
from mpclab_common.tracks.waypoint_projection_track import Planar2DWaypointTrack
from mpclab_common.tracks.radius_arclength_track import RadiusArclengthTrack


class CubicSplineTrack(Planar2DWaypointTrack):

    def __init__(self, width, cs, adjust_width=True, table_density=100, close_track=True):
        if isinstance(cs, CubicSpline):
            self.cs = cs
        else:
            self.cs = cs.tolist()  # numpy saves the spline object as a 0x0 array, this unpacks it

        self.width = width

        xy = np.array([self.cs(x) for x in self.cs.x])
        linpoints = np.arange(len(xy))

        self.N = len(xy)  # spline length (points)
        if close_track:
            self.X_int = CubicSpline(linpoints, xy[:, 0], bc_type='periodic')
            self.Y_int = CubicSpline(linpoints, xy[:, 1], bc_type='periodic')
        else:
            self.X_int = CubicSpline(linpoints, xy[:, 0], bc_type='not-a-knot')
            self.Y_int = CubicSpline(linpoints, xy[:, 1], bc_type='not-a-knot')
        self.dX = self.X_int.derivative(1)
        self.ddX = self.X_int.derivative(2)
        self.dY = self.Y_int.derivative(1)
        self.ddY = self.Y_int.derivative(2)

        self.track_length = self.cs.x[-1]
        self.track_width = width
        self.half_width = width / 2
        self.table_density = table_density
        self.generate_waypoints()

        if adjust_width:
            self.adjust_width()

        self.table = self.precompute_table(density=table_density)
        self.track_vars = ['sval', 'tval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']


        return

    def adjust_width(self):
        '''
        Adjustst the width to try to maximize the width of the track (without exceeding the maximum radius of curvature of the track)
        '''
        sl = np.linspace(0, self.track_length, 1000)
        kl = [self.get_curvature(s) for s in sl]
        kmax = max(kl)
        w_max = 1 / kmax * 2

        w = min(15, w_max)

        self.width = w
        self.track_width = w
        self.half_width = w / 2
        return

    def local_to_global_typed(self, data):
        x, y, psi = self.local_to_global((data.p.s, data.p.x_tran, data.p.e_psi))
        data.x = x
        data.y = y
        data.psi = psi
        return

    def local_to_global(self, coords, return_tangent=False):
        s, ey, epsi = coords

        xy = self.cs(s)

        ds = self.cs(s, 1)
        n = ds / np.linalg.norm(ds)
        n_t = np.array([n[1], -n[0]])

        xy = xy - ey * n_t

        x = xy[0]
        y = xy[1]

        psi_track = np.arctan2(n[1], n[0])
        psi = epsi + psi_track

        if not return_tangent:
            return (x, y, psi)
        else:
            return (x, y, psi, n[0], n[1])

    def plot_map(self, ax, pts_per_dist=None):
        if pts_per_dist is None:
            pts = 2000
        else:
            pts = self.track_length * pts_per_dist

        span_s = np.linspace(0, self.track_length, int(pts))
        middle = 0
        outer = self.width / 2
        inner = -outer

        middle_data = np.array([self.local_to_global((s, 0, 0)) for s in span_s])
        outer_data = np.array([self.local_to_global((s, outer, 0)) for s in span_s])
        inner_data = np.array([self.local_to_global((s, inner, 0)) for s in span_s])

        ax.plot(middle_data[:, 0], middle_data[:, 1], 'k--', linewidth=1)
        ax.plot(outer_data[:, 0], outer_data[:, 1], 'k')
        ax.plot(inner_data[:, 0], inner_data[:, 1], 'k')

        init_slope = self.cs(0, 1)
        start_line_vec = np.array([-init_slope[1], init_slope[0]]) / np.linalg.norm(init_slope)
        start_line_data = np.array([inner * start_line_vec, outer * start_line_vec])

        ax.plot(start_line_data[:, 0], start_line_data[:, 1], 'r', linewidth=1)
        ax.set_aspect('equal', 'box')

    def plot_map_qt(self, p, pts_per_dist= None):
        if pts_per_dist is None:
            pts = 2000
        else:
            pts = self.track_length * pts_per_dist

        span_s = np.linspace(0, self.track_length, int(pts))
        middle = 0
        outer = self.width / 2
        inner = -outer

        middle_data = np.array([self.local_to_global((s, 0, 0)) for s in span_s])
        outer_data = np.array([self.local_to_global((s, outer, 0)) for s in span_s])
        inner_data = np.array([self.local_to_global((s, inner, 0)) for s in span_s])

        p.plot(middle_data[:, 0], middle_data[:, 1], pen=pg.mkPen('k', width=1, dash=[4,2]))
        p.plot(outer_data[:, 0], outer_data[:, 1], pen=pg.mkPen('k', width=2))
        p.plot(inner_data[:, 0], inner_data[:, 1], pen=pg.mkPen('k', width=2))

        init_slope = self.cs(0, 1)
        start_line_vec = np.array([-init_slope[1], init_slope[0]]) / np.linalg.norm(init_slope)
        start_line_data = np.array([inner * start_line_vec, outer * start_line_vec])

        p.plot(start_line_data[:, 0], start_line_data[:, 1], pen=pg.mkPen('r', width=1))

        track_bbox = (np.amin(outer_data[:, 0]),
                      np.amin(outer_data[:, 1]),
                      np.amax(outer_data[:, 0]) - np.amin(outer_data[:, 0]),
                      np.amax(outer_data[:, 1]) - np.amin(outer_data[:, 1]))

        return track_bbox

    def plot_approximations(self, ax):
        sampling = 15  # number of points between samples
        coords = self.table[::sampling, self.track_vars.index('xtrack'):self.track_vars.index('ytrack')+1]
        phis = self.table[::sampling, self.track_vars.index('phitrack')]
        svals = self.table[::sampling, self.track_vars.index('sval')]
        tvals = self.table[::sampling, self.track_vars.index('tval')]
        cos_phi = self.table[::sampling, self.track_vars.index('cos(phi)')]
        sin_phi = self.table[::sampling, self.track_vars.index('sin(phi)')]
        gvals = self.table[::sampling, self.track_vars.index('g_upper')]

        len_indicator = 0.05
        ax.plot(self.table[:, 2], self.table[:, 3])
        ax.scatter(self.table[::sampling, 2], self.table[::sampling, 3], marker='o')
        for idx, coord in enumerate(coords):
            _x, _y = coord
            end = len_indicator * np.array([cos_phi[idx], sin_phi[idx]]) + np.array(coord)
            ax.plot([_x, end[0]], [_y, end[1]], color='r')

    def get_curvature(self, s):
        ds = self.cs(s, 1)
        d2s = self.cs(s, 2)

        k = (ds[0] * d2s[1] - ds[1] * d2s[0]) / np.power(ds[1] ** 2 + ds[0] ** 2, 1.5)
        return k

    def mod_s(self, s):
        while (s < 0):                  s += self.track_length
        while (s > self.track_length):  s -= self.track_length
        return s

    def eval_spline(self, t):
        return np.array([self.X_int(t), self.Y_int(t)])

    def calc_yaw(self, t):
        dx, dy = self.dX(t), self.dY(t)
        return np.arctan2(dy, dx)  # phi

    def fit(self):
        # using two full laps to account for horizon overshooting end of lap
        # giving better approximation near start/end
        npoints = 2 * 20 * self.N

        # Approximation stage for arc parameterization
        tvals = np.linspace(0, 2 * self.N, npoints + 1)
        coords = np.array([self.eval_spline(np.mod(t, self.N)) for t in tvals])
        distsr = [0]
        for idx in range(npoints):
            distsr.append(
                np.sqrt(np.sum(np.square(coords[idx, :] - coords[np.mod(idx + 1, npoints - 1), :])))
            )
        dists = np.cumsum(np.array(distsr))

        inverse_map = CubicSpline(dists, tvals)

        return inverse_map

    def precompute_table(self, density=100):
        # generate lookup
        inverse_map = self.fit()

        npoints = np.int(np.floor(2 * self.track_length * density))
        print(f"Precomputed table with {npoints} entries")
        svals = np.linspace(0, 2 * self.track_length, npoints)  # Using 2 laps to account for overshoot
        tvals = inverse_map(svals)

        #  entries : ['sval', 'tval', 'xtrack', 'ytrack', 'phitrack', 'cos(phi)', 'sin(phi)', 'g_upper', 'g_lower']
        table = []
        for idx in range(npoints):
            _x, _y = self.eval_spline(tvals[idx])
            phi = self.calc_yaw(tvals[idx])
            n = [-np.sin(phi), np.cos(phi)]
            g_upper = self.half_width + _x * n[0] + _y * n[1]
            g_lower = -self.half_width + _x * n[0] + _y * n[1]
            table.append(
                [svals[idx], tvals[idx],
                 _x, _y,
                 phi,
                 np.cos(phi), np.sin(phi),
                 g_upper, g_lower]
            )

        table = np.array(table)

        return table


def plot_tests():
    from mpclab_common.track import get_track
    from pyqtgraph.Qt import QtGui, QtCore

    track = get_track('Lab_Track_barc')
    from mpclab_common.tracks.generate_tracks import convert_radiusarclength_to_spline
    track = convert_radiusarclength_to_spline(track, pts_per_dist=None, table_density=100)

    print(f"Track Width: {track.track_width}")
    print(f"Track Length (spline): {track.track_length}")

    # # matplotlib
    fig = plt.figure()
    ax = plt.subplot(111)
    track.plot_map(ax)
    track.plot_approximations(ax)
    plt.show()

    # QT
    app = QtGui.QApplication([])
    pg.setConfigOptions(antialias=True, background='w')
    figsize = (750, 750)
    widget = pg.GraphicsLayoutWidget(show=True)
    widget.setWindowTitle('BARC Plotter')
    widget.resize(*figsize)
    l_xy = widget.addLayout()
    vb_xy = l_xy.addViewBox(lockAspect=True)
    p_xy = l_xy.addPlot(viewBox=vb_xy)
    p_xy.setDownsampling(mode='peak')
    p_xy.setLabel('left', 'Y', units='m')
    p_xy.setLabel('bottom', 'X', units='m')
    track_bbox = track.plot_map_qt(p_xy)  # plot the track once at the start and avoid deleting it.
    print(track_bbox)
    app.exec_()


if __name__ == "__main__":
    plot_tests()
