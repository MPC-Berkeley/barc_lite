
from mpclab_common.tracks.base_track import BaseTrack
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d import Axes3D
import os
import numpy as np
import csv
import pdb

'''
    Classes for approximately converting global coordinates to local coordinates using a buffer of waypoints
    
    planar2D: (ex: parametric_polynomial_track - ParametricPolynomialTrack)
        For any track that can curve in plane without banking.
        self.global_to_local must be implemented and be capable of returning a vector tangent to the path
        e.g.  x,y,psi = self.local_to_global((s,0,0))
              x,y,psi,xt,yt = self.local_to_global((s,0,0), reutrn_tangent = True)
              
        These are used to construct bases at each waypoint to approximate the longitudinal (s) and tagential (x_tran) deviation of the point
        each waypoint consists of [s,p,A,psi] where 
            s is the path length
            p is a np.array of [x,y]
            A is a np.array of [e_s, e_t] where e_s is a unit vector pointing in the direction of the path and e_t is orthogonal to e_s
                this obeys the coordinate system where e_s points forwards, e_z points up, and e_z // e_s X e_z 
            psi is the angular heading of the track
        
        when queried for a given [x,y] point:
            the closest waypoint (Euclidian distance) is found
            A and p are used to improve the estimate of s and estimate e_y
            the global heading and the track heading are used to calculate e_psi
            s, e_y, and e_psi are returned
    
    IMSTrack: 
        A special, self-contained track class that uses IMS track data to generate a track cable of banking and moving up/down        
    '''
    
class Planar2DWaypointTrack(BaseTrack):
    
    def generate_waypoints(self,n_segs = 5000):
        s_interp = np.linspace(0,self.track_length-1e-3,n_segs)
        self.waypoints = []
        for s in s_interp: 
            x, y, psi, xt, yt = self.local_to_global((s, 0, 0),return_tangent = True)
            p0 = np.array([x,y])
            tangent = np.array([xt,yt]) / np.sqrt(xt**2 + yt**2)
            A = np.array([[tangent[0],-tangent[1]],[tangent[1],tangent[0]]])
            self.waypoints.append([s, p0, A, psi])
        return
    

    def global_to_local_typed(self,data):
        xy_coord = (data.x.x, data.x.y, data.e.psi)
        cl_coord = self.global_to_local(xy_coord)
        if cl_coord:
            data.p.s = cl_coord[0]
            data.p.x_tran = cl_coord[1]
            data.p.e_psi = cl_coord[2]
            return 0
        return -1
        
            
    def global_to_local(self, coords):
        p_list = np.array([waypoint[1] for waypoint in self.waypoints])
        p_interp = np.array([coords[0],coords[1]])
        idx = np.argmin( np.linalg.norm(p_list - p_interp, axis=1) )
        p0 = self.waypoints[idx][1]
        
        A = self.waypoints[idx][2]
        
        b = np.linalg.inv(A) @ (p_interp-p0)

        
        s = self.waypoints[idx][0] + b[0]
        e_y = b[1]
        e_psi = self.fix_angle_range(coords[2] - self.waypoints[idx][3])
        #pdb.set_trace()
        
        return s, e_y, e_psi
        
    def local_to_global(self,s):
        raise NotImplementedError('')
        
    def get_curvature(self,s):
        raise NotImplementedError('')
    
    def get_bankangle(self,s):
        raise NotImplementedError('')
        
    def fix_angle_range(self,angle):
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle  


class IMSTrack():
    def __init__(self):
        IMS_folder = os.path.join(os.path.dirname(__file__), 'IMS')
        csvfile = os.path.join(IMS_folder, 'IMS_track_boundaries_coordinates.csv')
        
        path_lengths = []
        centers = []
        bases = []
        left_widths = []
        right_widths = []
        curvatures = []
        bankangles = []
        
        track_data = []
        with open(csvfile, newline = '') as csvfile:
            csv_reader = csv.reader(csvfile)
            row_num = 0
            for row in csv_reader:
                if row_num == 0: # skip the header row
                    row_num += 1
                    continue
                else:
                    track_data.append([float(s) for s in row])
        track_data = np.array(track_data)
        
        # calculate vector from left to right and find what fraction the centerline should be at (z = 0)
        t_vec = track_data[:,np.r_[1,3]] - track_data[:,np.r_[0,2]]
        t_vec_3d = track_data[:,np.r_[1,3,5]] - track_data[:,np.r_[0,2,4]]
        
        #alpha = track_data[:,5] / (track_data[:,5] - track_data[:,4])
        #alpha[np.isnan(alpha)] = 0.5   # z difference of 0 results in nan, replace these entries with 0.5 (midpoint)
        #alpha[np.isinf(alpha)] = 0.5   # left z of 0 results in inf, replace these too
        
        #center points (x,y,0)
        #pc = track_data[:,np.r_[0,2]] + (alpha*t_vec.T).T
        #pc = np.hstack([pc,np.zeros((pc.shape[0],1))])
        
        pc = track_data[:,np.r_[0,2,4]] + 0.5*t_vec_3d
        
        # left and right widths
        #wl = np.linalg.norm(t_vec, axis = 1) * alpha
        #wr = np.linalg.norm(t_vec, axis = 1) * (1-alpha)
        wl = np.linalg.norm(t_vec_3d, axis = 1) * 0.5
        wr = np.linalg.norm(t_vec_3d, axis = 1) * 0.5
        
        # bank angles
        bankangles = np.arctan2(track_data[:,5] - track_data[:,4], np.linalg.norm(t_vec, axis = 1)) 
        
        # headings
        N = pc.shape[0]
        delta_pc_unbiased = 0.5 * (pc[np.r_[1:N,0]] - pc[np.r_[N-1,0:N-1]])  # neighbor estimate of difference to avoid biasing derivative
        psi = np.arctan2(delta_pc_unbiased[:,1], delta_pc_unbiased[:,0])
        
        # path length
        delta_pc = pc[np.r_[1:N]] - pc[np.r_[0:N-1]]  
        ds = np.linalg.norm(delta_pc,axis = 1)
        s = np.concatenate([np.array([0]), np.cumsum(ds)])
        
        # unit vectors 
        ns = np.array([np.cos(psi), np.sin(psi), np.zeros((psi.shape[0]))])
        nt = np.array([-np.sin(psi)*np.cos(bankangles), np.cos(psi)*np.cos(bankangles), -np.sin(bankangles)])
        nz = np.array([-np.sin(psi)*np.sin(bankangles), np.cos(psi)*np.sin(bankangles),  np.cos(bankangles)])
        
        #pdb.set_trace()
        
        A = np.array([ns,nt,nz]).T
        
        # curvature method 1: local discretization
        ds_unbiased = np.linalg.norm(delta_pc_unbiased,axis = 1)
        dpds = (delta_pc_unbiased.T / ds_unbiased).T
        dxds = dpds[:,0]
        dyds = dpds[:,1]
        
        delta2pc_unbiased = 0.5 * (dpds[np.r_[1:N,0]] - dpds[np.r_[N-1,0:N-1]])
        d2pds2 = (delta2pc_unbiased.T / ds_unbiased).T
        d2xds2 = d2pds2[:,0]
        d2yds2 = d2pds2[:,1]
        curvature = ((dxds * d2yds2 - dyds * d2xds2).T / np.power(np.sqrt((dxds * dxds + dyds * dyds)),3.0)).T
    
        #track_length includes the distance from the last point back to the start
        self.track_length = np.max(s) + np.linalg.norm(pc[-1] - pc[0])
        self.min_left_width = np.min(wl)
        self.min_right_width = np.min(wr)
        self.waypoint_path_lengths = s
        self.waypoints = [[a,b,c,d,e,f,g,h] for (a,b,c,d,e,f,g,h) in zip(s,pc,A,psi,bankangles,curvature,wl,wr)]
        #pdb.set_trace()
        
        self.update_curvature_with_cubic_spline()
        self.zero_map_origin()
    
    def zero_map_origin(self,):
        ''' moves the track origin to (0,0,0)
            rotates the map so that the start postion is pointing at psi = 0 in the global frame
        '''
        offset = -self.waypoints[0][1]
        for waypoint in self.waypoints:
            waypoint[1] += offset

        rotation = -self.waypoints[0][3]
        rotation_matrix = np.array([[np.cos(rotation), np.sin(rotation),0],
                                   [-np.sin(rotation), np.cos(rotation),0],
                                   [0,0,1]])
        
        for waypoint in self.waypoints:
            waypoint[1] = rotation_matrix @ waypoint[1]
            waypoint[2] = rotation_matrix @ waypoint[2]
            waypoint[3] += rotation
        return
        
        
    def update_curvature_with_cubic_spline(self):
        s = [p[0]    for p in self.waypoints]
        x = [p[1][0] for p in self.waypoints]
        y = [p[1][1] for p in self.waypoints]
        s.append(self.track_length)
        x.append(x[0])
        y.append(y[0])
        
        csx = CubicSpline(s, x, bc_type='periodic')
        csy = CubicSpline(s, y, bc_type='periodic') 
        
        for i in range(len(self.waypoints)):
            s = self.waypoints[i][0]
            dxds = csx(s,1)
            dyds = csy(s,1)
            d2xds2 = csx(s,2)
            d2yds2 = csy(s,2)
            
            curvature = (dxds * d2yds2 - dyds * d2xds2) / np.power(np.sqrt((dxds * dxds + dyds * dyds)),3.0)
            self.waypoints[i][5] = curvature
    
    def local_to_global_typed(self,data):
        p,psi = self.local_to_global_typed((data.s,data.x_tran,data.e_psi))
        data.x = p[0]
        data.y = p[1]
        #data.z = p[2]
        data.psi = psi
        return
        
    def global_to_local_typed(self,data):
        #TODO(thomasfork): Figure out how to reconcile 3D position lookup with 2D state used elsewhere
            # currently the lack of a z state variable will incur error up to about 20% of the width when projecting at elevated locations
    
        s,ey,epsi = self.global_to_local((np.array([data.x,data.y,0]), data.psi))
        data.s = s
        data.x_tran = ey
        data.epsi = e_psi
        return
        
    def local_to_global(self,coords):
        ''' 
        converts track coordinates (s,ey,epsi) to global coordinates ((x,y,z), psi)
        '''    
        s,ey,epsi = coords
        idx = self.s2idx(s)
        
        p0 = self.waypoints[idx][1]
        A = self.waypoints[idx][2]
        
        p = p0 + A@ np.array([s - self.waypoints[idx][0], ey, 0])
        
        psi = epsi + self.waypoints[idx][3]
        
        return (p,psi)
            
    
    def global_to_local(self,coords):
        ''' 
        converts global coordinates ((x,y,z), psi) to track coordinates (s,ey,epsi) 
        '''    
        p,psi = coords
        p_list = np.array([waypoint[1] for waypoint in self.waypoints])
        idx = np.argmin( np.linalg.norm(p_list - p, axis=1) )
        
        p0 = self.waypoints[idx][1]
        
        A = self.waypoints[idx][2]
        
        b = np.linalg.inv(A) @ (p-p0)

        
        s = self.waypoints[idx][0] + b[0]
        s = self.mod_s(s)  # correct for small negative path lengths around the start
        e_y = b[1]
        e_psi = self.fix_angle_range(psi - self.waypoints[idx][3])
        
        return s, e_y, e_psi  
          
    def get_bank_angle(self,s):
        idx = self.s2idx(s)
        return self.waypoints[idx][4]
    
    def get_slope_angle(self,s):
        #TODO:
        return
    
    def get_curvature(self,s):
        idx = self.s2idx(s)
        return self.waypoints[idx][5]
        
    def get_left_width(self,s = None):
        if s is None:
            return self.min_left_width
        idx = self.s2idx(s)
        return self.waypoints[idx][6]
    
    def get_right_width(self,s = None):
        if s is None:
            return self.min_left_width
        idx = self.s2idx(s)
        return self.waypoints[idx][7]    
   
    def s2idx(self,s):
        return np.argmax(self.mod_s(s) < self.waypoint_path_lengths) -1 
        
    def mod_s(self,s):
        while (s < 0):                  s += self.track_length
        while (s > self.track_length):  s -= self.track_length
        return s          
                                  
    def fix_angle_range(self,angle):
        while angle > np.pi: angle -= 2*np.pi
        while angle < -np.pi: angle += 2*np.pi
        return angle  
        
    def plot_map(self,ax, n_segs = 1000):
        
        s_interp = np.linspace(0,self.track_length-1e-3,n_segs)
        
        
        line_c, line_l, line_r = [],[],[]
        
        for s in s_interp:
            line_c.append(self.local_to_global((s,0                  ,0))[0])
            line_l.append(self.local_to_global((s,self.get_left_width(s) ,0))[0])
            line_r.append(self.local_to_global((s,self.get_right_width(s),0))[0])
        
        line_c = np.array(line_c)   
        line_l = np.array(line_l)  
        line_r = np.array(line_r)  
    
                                     
        ax.plot_surface(np.array([line_l[:,0],line_c[:,0],line_r[:,0]]),
                        np.array([line_l[:,1],line_c[:,1],line_r[:,1]]),
                        np.array([line_l[:,2],line_c[:,2],line_r[:,2]]),color = 'blue', alpha = 0.6)
        
        ax.plot(line_c[:,0], line_c[:,1], line_c[:,2], '--k')
        ax.plot(line_l[:,0], line_l[:,1], line_l[:,2], 'k')
        ax.plot(line_r[:,0], line_r[:,1], line_r[:,2], 'k')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_zlim(-1,400)    
    
    '''def plot_map_opengl(self,n_segs = 1000):
        import pyqtgraph as pg
        import pyqtgraph.opengl as gl
        
        pg.mkQApp()
        view = gl.GLViewWidget()
        view.show()
        
        xgrid = gl.GLGridItem()
        ygrid = gl.GLGridItem()
        zgrid = gl.GLGridItem()
        view.addItem(xgrid)
        view.addItem(ygrid)
        view.addItem(zgrid)

        ## rotate x and y grids to face the correct direction
        xgrid.rotate(90, 0, 1, 0)
        ygrid.rotate(90, 1, 0, 0)

        ## scale each grid differently
        xgrid.scale(0.2, 0.1, 0.1)
        ygrid.scale(0.2, 0.1, 0.1)
        zgrid.scale(0.1, 0.2, 0.1)

        
        line_c, line_l, line_r = [],[],[]
        
        for s in s_interp:
            line_c.append(self.local_to_global((s,0                  ,0))[0])
            line_l.append(self.local_to_global((s,self.get_left_width(s) ,0))[0])
            line_r.append(self.local_to_global((s,self.get_right_width(s),0))[0])
        
        line_c = np.array(line_c)   
        line_l = np.array(line_l)  
        line_r = np.array(line_r)  '''
        
           
    def test_track_shape_parameters(self, n_segs = 1000):
        fig = plt.figure(figsize = (14,7))
        
        s_interp = np.linspace(0,self.track_length-1e-3,n_segs)
        
        
        line_c, line_b, line_wl, line_wr = [],[],[],[]
        
        for s in s_interp:
            line_c.append(self.get_curvature(s))
            line_b.append(self.get_bank_angle(s))
            line_wl.append(self.get_left_width(s))
            line_wr.append(self.get_right_width(s))
        
        line_c = np.array(line_c)   
        line_b = np.array(line_b)   
        line_wl = np.array(line_wl)   
        line_wr = np.array(line_wr)  
        
        plt.subplot(411)
        plt.title('Curvature')
        plt.plot(s_interp,line_c)
        plt.subplot(412)
        plt.title('Slope')
        plt.plot(s_interp,line_b)
        plt.subplot(413)
        plt.title('left width')
        plt.plot(s_interp,line_wl)
        plt.subplot(414)
        plt.title('right width')
        plt.plot(s_interp,line_wr)
        plt.xlabel('path length')
        plt.tight_layout()     
        plt.show()
    
    def test_plot_map(self):
        fig = plt.figure(figsize = (14,7))
        ax = fig.gca(projection='3d')
        self.plot_map(ax)
        plt.show()
    
    def test_projection_reconstruction_accuracy(self):
        fig = plt.figure(figsize = (14,7))
        
        n_segs = 1000
        s_interp = np.linspace(0,self.waypoints[-1][0],n_segs)
        errors = []
        for s in s_interp:
            e_y = np.random.uniform(-self.get_left_width(s), self.get_right_width(s))
            e_psi = np.random.uniform(-1,1)
            p, psi = self.local_to_global((s, e_y, e_psi))
            s_n, e_y_n, e_psi_n = self.global_to_local((p, psi))
            if (s_n - s)**2 > 5:
                pdb.set_trace()
            errors.append([(s-s_n)**2, (e_y - e_y_n)**2,(e_psi - e_psi_n)**2])
        for i in range(3):
            data = [err[i] for err in errors]
            plt.plot(data)
        plt.legend(('s','e_y','e_psi'))
        plt.title('Reconstruction squared errors')
        plt.show()
        
        
        return
        
    def all_visual_tests(self):
        self.test_projection_reconstruction_accuracy()
        self.test_track_shape_parameters()
        self.test_plot_map()      
                        
def main():
    track = IMSTrack()
    track.all_visual_tests()
    return

if __name__ == '__main__':
    main()                    
                       
