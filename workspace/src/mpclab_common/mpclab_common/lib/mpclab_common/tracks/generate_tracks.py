import numpy as np
import os

def get_save_folder():
    return os.path.join(os.path.dirname(__file__), 'track_data')

def generate_curvature_and_path_length_track(filename, track_width, cl_segs, slack):

    save_folder = get_save_folder()
    os.makedirs(save_folder, exist_ok=True)

    save_path = os.path.join(save_folder, filename)
    np.savez(save_path, save_mode = 'radius_and_arc_length', track_width = track_width, cl_segs = cl_segs, slack = slack)
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
    generate_LTrack_barc()
    
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
