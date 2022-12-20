import numpy as np
import pickle
import pdb
import csv
import os
import urllib.request
import json
from matplotlib import pyplot as plt

from scipy import optimize
from scipy.interpolate import CubicSpline

from mpclab_common.tracks.poly_raceline_tracks import Map2
from mpclab_common.tracks.cubic_spline_track import CubicSplineTrack
from mpclab_common.tracks.radius_arclength_track import RadiusArclengthTrack

def get_save_folder():
    return os.path.join(os.path.dirname(__file__), 'track_data')

def generate_curvature_and_path_length_track(filename, track_width, cl_segs, slack):

    save_folder = get_save_folder()
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, save_mode = 'radius_and_arc_length', track_width = track_width, cl_segs = cl_segs, slack = slack)
    return

def generate_cubic_spline_track(filename, track_width, cs):
    save_folder = get_save_folder()
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, save_mode = 'cubic_spline', track_width = track_width, cs = cs)

def generate_straight_track():
    track_width = 1.0
    slack = 0.45

    length = 10.0
    cl_segs = np.array([length, 0]).reshape((1,-1))

    generate_curvature_and_path_length_track('Straight_Track', track_width, cl_segs, slack)
    print('Generated Straight_Track')
    return

def generate_LTrack_barc():
    track_width = 1.1
    slack     = 0.3

    ninety_radius_1     = ((1.5912+0.44723)/2 + (1.5772+0.46504)/2)/2 
    ninety_radius_2     = ((0.65556 + 1.12113/2) + (0.6597 + 1.13086/2))/2

    oneeighty_radius_1  = (1.171 + 1.1473/2 + 1.1207/2)/2
    oneeighty_radius_2  = (1.3165 + 1.15471/2 + 1.12502/2)/2

    straight_1          = 2.401 - 0.15
    straight_2          = 1.051 - 0.15
    straight_3          = 0.450 - 0.3
    straight_4          = 2*oneeighty_radius_1 + ninety_radius_1 + straight_3 - ninety_radius_2 #2.5515
    straight_5          = np.abs(straight_1 - straight_2 - ninety_radius_1 - 2*oneeighty_radius_2 + ninety_radius_2)

    cl_segs = np.array([[straight_1,                    0],
                        [np.pi*oneeighty_radius_1,      oneeighty_radius_1],
                        [straight_2,                    0],
                        [np.pi/2*ninety_radius_1,       -ninety_radius_1],
                        [straight_3,                    0],
                        [np.pi*oneeighty_radius_2,      oneeighty_radius_2],
                        [straight_4,                    0],
                        [np.pi/2*ninety_radius_2,       ninety_radius_2],
                        [straight_5,                    0]])

    generate_curvature_and_path_length_track('L_track_barc', track_width, cl_segs, slack)
    print('Generated L_track_barc')
    return

# def generate_Lab_track():
#     track_width = 0.75
#     slack     = 0.45
#
#     straight_1 = 2.3364
#     straight_2 = 1.9619
#     straight_3 = 0.5650
#     straight_4 = 0.5625
#     straight_5 = 1.3802
#     straight_6 = 0.8269
#     total_width = 3.4779
#
#     ninety_radius = 0.758 # (3.49 - 2.11)/2 # Radius of 90 deg turns
#
#     thirty_secant = 0.5*((straight_6 + straight_1)-(straight_3 + straight_5 + straight_4*np.cos(np.pi/6)))/np.cos(15*np.pi/180)
#     thirty_radius = 0.5*(thirty_secant)/np.cos(75*np.pi/180) # Radius of 30 deg turns
#
#     oneeighty_radius = 0.5*(total_width-straight_4*np.sin(np.pi/6)-2*thirty_secant*np.cos(75*np.pi/180))
#
#     cl_segs = np.array([[straight_1,                0], #[2.375, 0],
#                         [np.pi/2 * ninety_radius,   ninety_radius],
#                         [straight_2,                0],  # [2.11, 0],
#                         [np.pi/2 * ninety_radius,   ninety_radius],
#                         [straight_3,                0], # [0.62, 0],
#                         [np.pi/6 * thirty_radius,   thirty_radius],
#                         [straight_4,                0], # [0.555, 0],
#                         [np.pi/6 * thirty_radius,   -thirty_radius],
#                         [straight_5,                0], #[1.08, 0],
#                         [np.pi * oneeighty_radius,  oneeighty_radius],
#                         [straight_6,                0]]) #[0.78, 0]])
#
#     generate_curvature_and_path_length_track('Lab_Track_barc', track_width, cl_segs, slack)
#     print('Generated Lab_Track_barc')
#     return

def generate_Lab_track():
    track_width = 1.19
    slack     = 0.45
    origin = [3.377, 0.9001]
    pt_before_circle_inside = [2.736, 2.078]
    pt_after_half_circle_inside = [1.211,  2.078]
    end_2nd_straight_inside = [1.300, -0.396]
    last_pt = [2.8166, -0.396]
    straight_1 = pt_before_circle_inside[1] - origin[1] - .225
    straight_2 = -end_2nd_straight_inside[1] + pt_after_half_circle_inside[1] - .45
    straight_final = straight_2 - straight_1
    radius_first = (-pt_after_half_circle_inside[0] + pt_before_circle_inside[0] + track_width)/2

    cl_segs = np.array([[straight_1,                0],
                        [np.pi * radius_first,  radius_first],
                        [straight_2,                0],
                        [np.pi * radius_first,  radius_first],
                        [straight_final, 0]])

    generate_curvature_and_path_length_track('Lab_Track_barc', track_width, cl_segs, slack)
    print('Generated Lab_Track_barc')
    return

def generate_Lab_track_old():
    track_width = 1.19
    slack     = 0.45

    straight_1 = 1.75 - 0.20
    straight_2 = 2.55 - 0.38
    straight_final = straight_2 - straight_1
    radius_first = (2.08+0.6)/2.0

    cl_segs = np.array([[straight_1,                0],
                        [np.pi * radius_first,  radius_first],
                        [straight_2,                0],
                        [np.pi * radius_first,  radius_first],
                        [straight_final, 0]])

    generate_curvature_and_path_length_track('Lab_Track_barc_old', track_width, cl_segs, slack)
    print('Generated Lab_Track_barc_old')
    return

def generate_Monza_track():
    track_width = 1.5
    slack     = 0.45 # Don't change?

    straight_01 = np.array([[3.0, 0]])

    enter_straight_length = 0.5
    curve_length = 3
    curve_swept_angle = np.pi - np.pi/40
    s = 1
    exit_straight_length = 2.2
    curve_11 = np.array([
                        [curve_length, s * curve_length / curve_swept_angle],
                        [exit_straight_length, 0]])


    # WACK CURVE

    enter_straight_length = 0.2
    curve_length = 0.8
    curve_swept_angle = np.pi / 4
    s = -1
    exit_straight_length = 0.4
    curve_10 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    enter_straight_length = 0.05
    curve_length = 1.2
    curve_swept_angle = np.pi / 4
    s = 1
    exit_straight_length = 0.4
    curve_09 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    enter_straight_length = 0.05
    curve_length = 0.8
    curve_swept_angle = np.pi / 4
    s = -1
    exit_straight_length = 2.5
    curve_08 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])


    # Curve mini before 07
    enter_straight_length = 0
    curve_length = 1.0
    curve_swept_angle = np.pi / 12
    s = -1
    exit_straight_length = 1.5
    curve_before_07 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Curve 07
    enter_straight_length = 0
    curve_length = 1.2
    curve_swept_angle = np.pi / 3
    s = 1
    exit_straight_length = 1.5
    curve_07 = np.array([ [curve_length, s * curve_length / curve_swept_angle],
                                [exit_straight_length, 0]])

    # Curve 06
    enter_straight_length = 0
    curve_length = 1.5
    curve_swept_angle = np.pi / 2
    s = 1
    exit_straight_length = 2.0
    curve_06 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Chicane 05
    enter_straight_length = 0
    curve1_length = 0.4
    s1, s2 = 1, -1
    curve1_swept_angle = np.pi/8
    mid_straight_length = 1.0
    curve2_length = 0.4
    curve2_swept_angle = np.pi/8
    exit_straight_length = 1.0 + np.abs(np.sin(-1.6493361431346418)*(0.4299848245548139+0.0026469133545887783))


    chicane_05 = np.array([
                      [curve1_length, s1 * curve1_length / curve1_swept_angle],
                      [mid_straight_length, 0],
                      [curve2_length, s2 * curve2_length / curve2_swept_angle],
                      [exit_straight_length, 0]])

    # Curve 03
    enter_straight_length = 0.0
    curve_length = 4.0
    curve_swept_angle = np.pi / 2 + np.pi/16
    s = 1
    exit_straight_length = 2.0
    curve_03 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Curve 02
    enter_straight_length = 0.0
    curve_length = 1.0
    curve_swept_angle = np.pi / 10
    s = -1
    exit_straight_length = 1.0
    curve_02 = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])

    # Final curve
    curve_length = 1.0
    curve_swept_angle = -np.pi / 10 + 0.11780972450961658
    exit_straight_length = 1.0 - 1.26433096e-01 + 0.0002070341780330276 + 0.00021382215942933325 + 1.6293947847880575e-05- 0.00023011610727452503
    s = -1
    curve_Final = np.array([
                         [curve_length, s * curve_length / curve_swept_angle],
                         [exit_straight_length, 0]])


    cl_segs = []
    cl_segs.extend(straight_01)
    cl_segs.extend(curve_11)
    cl_segs.extend(curve_10)
    cl_segs.extend(curve_09)
    cl_segs.extend(curve_08)
    cl_segs.extend(curve_before_07)
    cl_segs.extend(curve_07)
    cl_segs.extend(curve_06)
    cl_segs.extend(chicane_05)
    cl_segs.extend(curve_03)
    cl_segs.extend(curve_02)
    cl_segs.extend(curve_Final)


    generate_curvature_and_path_length_track('Monza_Track', track_width, cl_segs, slack)
    print('Generated Monza_Track')
    return

def generate_Circle_track():
    track_width = 1.0
    slack     = 0.45
    radius=4.5
    cl_segs = np.array([[np.pi*radius, radius],
                                [np.pi*radius, radius]])

    generate_curvature_and_path_length_track('Circle_Track_barc', track_width, cl_segs, slack)
    print('Generated Circle_Track_barc')
    return

def generate_Oval_track():
    straight_len=2.0
    curve_len=5.5
    track_width=1.0
    slack=0.45
    cl_segs = np.array([[straight_len/2, 0],
                                [curve_len, curve_len/np.pi],
                                [straight_len, 0],
                                [curve_len, curve_len/np.pi],
                                [straight_len/2, 0]])

    generate_curvature_and_path_length_track('Oval_Track_barc', track_width, cl_segs, slack)
    print('Generated Oval_Track_barc')
    return

def generate_Indy_track():
    straight_len=1006
    short_len = 201
    curve_len= 402
    track_width=15
    slack=0.45
    cl_segs = np.array([[straight_len/2, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [short_len, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [straight_len, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [short_len, 0],
                                [curve_len, 2*curve_len/np.pi],
                                [straight_len/2, 0]])

    generate_curvature_and_path_length_track('Indy_Track', track_width, cl_segs, slack)
    print('Generated Indy_Track')
    return



#(thomasfork) - code that was considered for fitting arc segments to a sequence of points, does not currently work but may be revisited
'''def fit_circle(x,y):

    #Fits a circle to an array of x and y points, returns center point, radius, and least squares residual normalized by # of points



    def calc_R(xc,yc):
        return np.sqrt((x-xc)**2 + (y-yc)**2)

    def f(c):
        Ri = calc_R(*c)
        return Ri - Ri.mean()

    def Df(c):
        xc, yc     = c
        df_dc    = np.empty((len(c), x.size))
        Ri = calc_R(xc, yc)
        df_dc[0] = (xc - x)/Ri                   # dR/dxc
        df_dc[1] = (yc - y)/Ri                   # dR/dyc
        df_dc    = df_dc - df_dc.mean(axis=1)[:, np.newaxis]


        return df_dc

    center_estimate = np.mean(x), np.mean(y)
    center, _ = optimize.leastsq(f, center_estimate, Dfun=Df, col_deriv=True)

    xc, yc = center
    Ri           = calc_R(*center)
    R            = Ri.mean()
    residual     = sum((Ri - R)**2) #/ len(x)
    #pdb.set_trace()

    return xc, yc, R, residual

def fit_circle_ends(x,y, psi0):
    ps = np.array([x[0],y[0]])
    pf = np.array([x[-1],y[-1]])
    psif = np.arctan2(y[-1] - y[-2], x[-1] - x[-2])

    pm = (ps + pf)/2

    d = np.array([pf[1]-ps[1], ps[0] - pf[0]])  # vector between pm and pc

    n = np.array([np.cos(psi0),np.sin(psi0)])

    #derives from pc = alpha*d + pm  and np.dot(pc-ps, n) == 0
    alpha = (-np.dot(pm,n) + np.dot(ps,n)) / np.dot(d,n)

    pc = alpha*d + pm

    R= np.linalg.norm(pc-ps)

    if np.cross(pf-pc,ps-pc) > 0: R *= -1

    arc_angle = np.arccos(np.clip(np.dot(pf-pc,ps-pc)  /\
                          np.linalg.norm(pf-pc) /\
                          np.linalg.norm(ps-pc), -1,1))

    while psi0 > np.pi : psi0 -= np.pi*2
    while psi0 < -np.pi: psi0 += np.pi*2

    residual = (psif - psi0 - arc_angle) **2


    return R, arc_angle, residual


def path_to_curve_path(path, tol):

    #Fits a sequence of arcs to a path of x,y coordinates.
    #'tol' is a tolerance used for the fit_circle

    #Fitting is done by repeatedly fitting a circle to the next N points and increasing N until the least squares fit error exceeds 'tol'
    #A circle is then fit to the start and end points.

    #current_path_position and psi are updated along the way to ensure that the reconstructed track will have the right shape


    num_pts = path.shape[0]
    num_pts_covered = 0
    print('converting to curve (%d/%d)'%(num_pts_covered,num_pts), end = '\r')

    arc_lengths = []
    radii = []

    psi = np.arctan2(path[1,1]-path[0,1], path[1,0] - path[0,0])  # heading of the path

    current_path_position = path[0,:]

    while num_pts_covered < num_pts-1:

        #estimate how large a window can be fit with a circle
        window =  2
        residual = -1
        while residual < tol:
            #xc, yc, R, residual = fit_circle(path[num_pts_covered: num_pts_covered + window,0],
            #                                 path[num_pts_covered: num_pts_covered + window,1])
            #pdb.set_trace()
            _,_,residual = fit_circle_ends(path[num_pts_covered: num_pts_covered + window,0],path[num_pts_covered: num_pts_covered + window,1],psi)
            if num_pts_covered + window == num_pts-1:
                break
            else:
                window += 1

        print(residual)


        # the exact fit should be from start to finish with the previous tangent vector
        #ps = path[num_pts_covered,:]

        ps = current_path_position
        pf = path[num_pts_covered + window, :]
        pm = (ps + pf)/2

        d = np.array([pf[1]-ps[1], ps[0] - pf[0]])  # vector between pm and pc

        n = np.array([np.cos(psi),np.sin(psi)])

        #derives from pc = alpha*d + pm  and np.dot(pc-ps, n) == 0
        alpha = (-np.dot(pm,n) + np.dot(ps,n)) / np.dot(d,n)

        pc = alpha*d + pm

        R= np.linalg.norm(pc-ps)

        if np.cross(pf-pc,ps-pc) > 0: R *= -1

        arc_angle = np.arccos(np.clip(np.dot(pf-pc,ps-pc)  /\
                              np.linalg.norm(pf-pc) /\
                              np.linalg.norm(ps-pc), -1,1))

        arc_lengths.append(np.abs(arc_angle*R))
        radii.append(R)


        # update the current position of the reconsructed path
        current_path_center = current_path_position + radii[-1] * np.array([-np.sin(psi), np.cos(psi)])
        theta = abs(arc_angle) * np.sign(R)
        current_path_position = current_path_center + radii[-1] * np.array([np.sin(psi + theta), -np.cos(psi + theta)])


        psi += theta
        num_pts_covered += window
        #print('converting to curve (%d/%d)'%(num_pts_covered,num_pts), end = '\r')
        #print(window)

    print('\nfinished curve conversion')

    cl_segs = np.array([arc_lengths,radii]).T

    return cl_segs'''



def generate_F1_tracks():
    maps = ['Indy','LShape','Dubai','IT','HU']
    for map in maps:

        m = Map2(10,map)
        width = 0.1

        coefs = m.coefs
        xy = coefs[:,-1].reshape(-1,2)
        xy = np.vstack([xy,xy[0]])
        xy = xy - xy[0]
        rot = -np.arctan2(xy[1,1] - xy[0,1], xy[1,0] - xy[0,0])
        xy = xy @ np.array([[np.cos(rot), np.sin(rot)],[-np.sin(rot),np.cos(rot)]])


        N = xy.shape[0]
        delta_xy = xy[np.r_[1:N]] - xy[np.r_[0:N-1]]
        ds = np.linalg.norm(delta_xy,axis = 1)
        sl = np.concatenate([np.array([0]), np.cumsum(ds)])
        cs = CubicSpline(sl,xy,bc_type='periodic')

        generate_cubic_spline_track('F1_'+map, width, cs)

        print('Generated F1 track: ' + map)


#TODO(thomasfork) - WIP to generate F1 tracks from source
def generate_F1_json_tracks():
    path = os.path.join(os.path.dirname(__file__), 'F1_source_data')
    filename = 'it-2019.geojson'
    path = os.path.join(path,filename)
    with open(path) as f:
        datastring = f.read()
    data = json.loads(datastring)
    xycoords = []
    cdata = []
    for feature in data['features']:
        print(feature['id'])
        id = int(feature['id'][feature['id'].index('/')+1:])
        if id == 38168747:
            continue
        if feature['geometry']['type'] == 'LineString':
            for coordinate in feature['geometry']['coordinates']:
                xycoords.append(np.array(coordinate))
                cdata.append(id % 100)
        elif feature['geometry']['type'] == 'Point':
            xycoords.append(np.array(feature['geometry']['coordinates']))
            cdata.append(id % 100)
        else:
            raise NotImplementedError('Unrecognized feature type: %s'%feature['geometry']['type'])

    xycoords = np.array(xycoords)
    xycoords -= xycoords[0,:]
    xycoords *= np.pi/180 * 6371000


    plt.subplot(211)
    plt.scatter(xycoords[:,0],xycoords[:,1],c = cdata)
    plt.colorbar()
    plt.subplot(212)
    plt.plot(xycoords[:,0],'o')
    plt.show()

def convert_radiusarclength_to_spline(track : RadiusArclengthTrack, pts_per_dist=None, adjust_width=False, table_density=None, close_track=True) -> CubicSplineTrack:
    if pts_per_dist is None:
        pts = 2000
    else:
        pts = track.track_length * pts_per_dist
    if close_track:
        span_s = np.linspace(0, track.track_length, int(pts))
    else:
        span_s = np.linspace(0, track.track_length-0.001, int(pts))
    xycoords = np.array([track.local_to_global((s, 0, 0))[:2] for s in span_s])  # center-line
    if close_track:
        cs = CubicSpline(span_s, xycoords, bc_type='periodic')
    else:
        cs = CubicSpline(span_s, xycoords, bc_type='not-a-knot')

    cs_track = CubicSplineTrack(track.track_width, cs, adjust_width=adjust_width, table_density=table_density, close_track=close_track)
    return cs_track

def normsort(xycoords):
    sortedcoords = np.zeros(xycoords.shape)
    sorted_idx = []
    for i in range(xycoords.shape[0]):
        if i == 0:
            sortedcoords[i,:] = xycoords[i]
            sorted_idx.append(i)
        else:
            idx = np.argsort(np.linalg.norm(xycoords - sortedcoords[i-1], axis = 1)).tolist()
            while idx[0] in sorted_idx:
                idx.pop(0)
            sortedcoords[i,:] = xycoords[idx[0]]
            sorted_idx.append(idx[0])
    return sortedcoords


def main():
    # generate_straight_track()
    # generate_F1_json_tracks()
    generate_LTrack_barc()
    generate_Lab_track()
    generate_Lab_track_old()
    # generate_Circle_track()
    # generate_Oval_track()
    # generate_Indy_track()
    # generate_F1_tracks()
    # generate_Monza_track()
    
    from mpclab_common.track import get_track
    import matplotlib.pyplot as plt
    import pdb

    track = get_track('L_track_barc')
    fig = plt.figure()
    ax = plt.gca()
    track.plot_map(ax)
    ax.set_aspect('equal')
    plt.show()
    return


if __name__ == '__main__':
    main()
