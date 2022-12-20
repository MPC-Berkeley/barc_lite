import numpy as np

from mpclab_common.tracks.radius_arclength_track import RadiusArclengthTrack
from mpclab_common.tracks.cubic_spline_track import CubicSplineTrack
from mpclab_common.tracks.generate_tracks import get_save_folder

import pdb
import os

def get_available_tracks():
    save_folder = get_save_folder()
    return os.listdir(save_folder)
    
def get_track(track_file):
    if not track_file.endswith('.npz'): track_file += '.npz'
    
    if track_file not in get_available_tracks():
        raise ValueError('Chosen Track is unavailable: %s\nlooking in:%s\n Available Tracks: %s'%(track_file, 
                        os.path.join(os.path.dirname(__file__), 'tracks', 'track_data'),
                        str(get_available_tracks())))

    save_folder = get_save_folder()
    load_file = os.path.join(save_folder,track_file)
    
    npzfile = np.load(load_file, allow_pickle = True)
    if npzfile['save_mode'] == 'radius_and_arc_length':
        track = RadiusArclengthTrack()
        track.initialize(npzfile['track_width'], npzfile['slack'], npzfile['cl_segs'])
    elif npzfile['save_mode'] == 'cubic_spline':
        track = CubicSplineTrack(npzfile['track_width'],npzfile['cs'])
    else:
        raise NotImplementedError('Unknown track save mode: %s'%npzfile['save_mode'])
        
    return track   
